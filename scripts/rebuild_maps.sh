#!/usr/bin/env bash
# Re-download the datasets and retrain the 19 v2 maps of the RA-L letter, keeping ckpts/ as it is.
# Per scene, as in the 2026-11-03/04 build (docs/reports/2026-11-03-v2-maps-both-datasets, rebuild_v2.sh, v2b, v2e):
#   1. fit the candidate that build selected (base or 640x480; v2_select_7scenes.txt, v2b_select_12scenes.txt) from the
#      starting head ckpts/edl_head_sel.pt (7Scenes) or ckpts/rb0/edl_head_12scenes_base.pt (12Scenes),
#   2. re-apply the paper's head ckpts/v2/<dataset>/head_sel.pt,
#   3. fine-tune at 640x480 -> maps/v2/<dataset>/<scene>_r1.pt,
#   4. fit the 64-d descriptor field -> maps/v2/<dataset>/<scene>_r1_lf64.pt (fused CPU Adam for the two largest maps).
# The encoders in ckpts/v2 were trained on the deleted maps, and training is stochastic, so a rebuilt map is close to,
# not equal to, the paper's. Check with `eval` (new folders results/rebuild_*, the paper's results/v2_* stay untouched);
# `encoders` retrains the encoders on the rebuilt maps into ckpts/v2_rebuild without touching ckpts/v2.
# Time on an RTX 4070 Ti with a free CPU (step times of the 2026-11-03/04 build): about 10 h for `maps`, most of it the
# descriptor field; `encoders` adds about 3 h; `eval` about 15 h. Download 34 GB (+ 2.75 GB MASt3R checkpoint, which
# `eval` needs), final maps 31 GB. The baselines (ACE, DSAC*, Marepo, GSplatLoc) are not restored here; their results
# stay in results/ (sources: docs/reports/2026-11-22-storage-cleanup/logs/deleted_third_party.txt).
#   usage: bash scripts/rebuild_maps.sh check|data|check-data|maps|encoders|eval|all    (all = check, data, maps)
#          DRY=1 bash scripts/rebuild_maps.sh maps                  # print the commands only
#          ENC=ckpts/v2_rebuild bash scripts/rebuild_maps.sh eval   # evaluate with the retrained encoders
# Every step skips an output that exists, so a stage can be rerun after an interruption. Logs: maps/v2/logs/.
set -u
cd "$(dirname "$0")/.."
VENV=${VENV:-$HOME/venvs/uags}; CUDA=${CUDA_HOME:-$HOME/.local/cuda-12.8}
export PATH=$VENV/bin:$CUDA/bin:$PATH CUDA_HOME=$CUDA LD_LIBRARY_PATH=$CUDA/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAST3R_ROOT=${MAST3R_ROOT:-$(cd ../.. && pwd)/third_party}; export MAST3R_CKPT=${MAST3R_CKPT:-$MAST3R_ROOT/mast3r_ckpt}
MAPS_ROOT=${MAPS_ROOT:-maps/v2}; ENC=${ENC:-ckpts/v2}
DEPTH="--depth-weight 0.2 --depth-scale 1.0"
HQ="--fit-width 640 --fit-height 480 --iters 20000 --fit-stride 5 --views-uint8"
URL7=http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip
URL12=http://graphics.stanford.edu/projects/reloc/data/apt1.zip
PGT_REPO=https://github.com/tsattler/visloc_pseudo_gt_limitations
MAST3R_HF=naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
PGT_COMMIT=$(sed -n 's/^# visloc_pseudo_gt_limitations commit //p' scripts/data_manifest.txt)
PICKS_7scenes="chess:hq fire:hq heads:base office:base pumpkin:base redkitchen:hq stairs:base"
PICKS_12scenes="apt1/kitchen:hq apt1/living:base apt2/bed:hq apt2/kitchen:base apt2/living:hq apt2/luke:hq office1/gates362:base office1/gates381:hq office1/lounge:base office1/manolis:hq office2/5a:hq office2/5b:base"
L=$MAPS_ROOT/logs; mkdir -p $L; C=$L/chain.log
log() { echo "$* $(date '+%F %T')" | tee -a $C; }
u() { echo $1 | tr '/' '_'; }
cfg() {  # dataset -> ROOT PGT HEAD0 M K PICKS
  if [ $1 = 7scenes ]; then ROOT=data/7scenes_source; HEAD0=ckpts/edl_head_sel.pt; PICKS=$PICKS_7scenes
  else ROOT=data/12scenes_source; HEAD0=ckpts/rb0/edl_head_12scenes_base.pt; PICKS=$PICKS_12scenes; fi
  PGT=data/visloc_pseudo_gt_limitations/pgt/sfm/$1; M=$MAPS_ROOT/$1; K=ckpts/v2/$1; mkdir -p $M
}
step() {  # NAME OUTPUT command...   (skip if OUTPUT exists; DRY=1 prints only)
  local NAME=$1 OUT=$2; shift 2
  [ -e "$OUT" ] && return 0
  if [ -n "${DRY:-}" ]; then echo "+ $*"; return 0; fi
  local T0=$(date +%s); "$@" > $L/$NAME.log 2>&1; local rc=$?
  log "$NAME rc=$rc $(( ($(date +%s) - T0) / 60 )) min"; return $rc
}

check() {
  local ok=1 f DS
  python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && echo "ok   CUDA torch in $VENV" || { echo "FAIL CUDA torch in $VENV"; ok=0; }
  for f in FastSAM-s.pt "$MAST3R_ROOT/mast3r" "$MAST3R_ROOT/dust3r" scripts/data_manifest.txt ckpts/edl_head_sel.pt ckpts/rb0/edl_head_12scenes_base.pt \
           ckpts/v2/7scenes/head_sel.pt ckpts/v2/12scenes/head_sel.pt; do
    [ -e "$f" ] && echo "ok   $f" || { echo "FAIL missing $f"; ok=0; }; done
  for DS in 7scenes 12scenes; do echo "ok   $(ls ckpts/v2/$DS/*_encoders.pt 2>/dev/null | wc -l) encoder sets in ckpts/v2/$DS (kept)"; done
  [ -e "$MAST3R_CKPT/model.safetensors" ] && echo "ok   $MAST3R_CKPT" || echo "note $MAST3R_CKPT missing; \`data\` downloads it from Hugging Face ($MAST3R_HF)"
  curl -sfIL --max-time 30 "https://huggingface.co/$MAST3R_HF/resolve/main/config.json" > /dev/null && echo "ok   reachable huggingface.co/$MAST3R_HF" || { echo "FAIL unreachable huggingface.co/$MAST3R_HF"; ok=0; }
  for f in $URL7 $URL12; do curl -sfIL --max-time 30 "$f" > /dev/null && echo "ok   reachable $f" || { echo "FAIL unreachable $f"; ok=0; }; done
  git ls-remote "$PGT_REPO" > /dev/null 2>&1 && echo "ok   reachable $PGT_REPO" || { echo "FAIL unreachable $PGT_REPO"; ok=0; }
  echo "free disk: $(df -h . | tail -n 1 | awk '{print $4}') (needed: 34 GB data + 31 GB maps)"
  [ $ok = 1 ]
}
check_data() {  # file count and bytes of every scene against scripts/data_manifest.txt
  local bad=0 p n b n2 b2
  while read -r p n b; do
    case $p in \#*|"") continue;; esac
    [ -d "$p" ] || { echo "MISSING $p"; bad=1; continue; }
    n2=$(find "$p" -type f | wc -l); b2=$(find "$p" -type f -printf "%s\n" | awk '{s+=$1} END {printf "%.0f", s}')
    [ "$n2" = "$n" ] && [ "$b2" = "$b" ] && echo "ok   $p" || { echo "DIFF $p: $n2 files / $b2 bytes, expected $n / $b"; bad=1; }
  done < scripts/data_manifest.txt
  return $bad
}
data() {
  bash scripts/download_7scenes.sh && bash scripts/download_12scenes.sh || return 1
  [ -d data/visloc_pseudo_gt_limitations/.git ] || git clone -q $PGT_REPO data/visloc_pseudo_gt_limitations || return 1
  git -C data/visloc_pseudo_gt_limitations checkout -q $PGT_COMMIT || return 1
  [ -e "$MAST3R_CKPT/model.safetensors" ] || python -c "from huggingface_hub import snapshot_download; snapshot_download('$MAST3R_HF', local_dir='$MAST3R_CKPT')" || return 1
  check_data
}
maps() {
  local DS E SC V U FIT EXTRA LF
  for DS in 7scenes 12scenes; do cfg $DS
    for E in $PICKS; do SC=${E%:*}; V=${E#*:}; U=$(u $SC)
      if [ $V = hq ]; then FIT=$M/${U}_hq.pt; EXTRA=$HQ; else FIT=$M/$U.pt; EXTRA=""; fi
      if [ ! -e $M/${U}_r1.pt ]; then
        step ${DS}_${U}_fit $FIT python scripts/train_map.py --root $ROOT --dataset $DS --scene $SC --sfm-pose-dir $PGT \
          --head-ckpt $HEAD0 --out $FIT --psnr-frames 20 $DEPTH $EXTRA || return 1
        step ${DS}_${U}_head $L/${DS}_${U}_head.done bash -c "python scripts/reapply_head.py --head $K/head_sel.pt --maps $FIT && touch $L/${DS}_${U}_head.done" || return 1
        step ${DS}_${U}_ft $M/${U}_r1.pt python scripts/finetune_map.py --root $ROOT --dataset $DS --scene $SC --sfm-pose-dir $PGT \
          --map $FIT --out $M/${U}_r1.pt --psnr-frames 20 $DEPTH || return 1
      fi
      LF="python scripts/analysis/train_locfeat.py --root $ROOT --dataset $DS --scene $SC --sfm-pose-dir $PGT --map $M/${U}_r1.pt --out $M/${U}_r1_lf64.pt --pca-dim 64 --feats-only"
      case $U in
        office1_gates381|office1_manolis) step ${DS}_${U}_field $M/${U}_r1_lf64.pt $LF --cpu-opt-state || return 1;;   # >2M Gaussians
        *) step ${DS}_${U}_field $M/${U}_r1_lf64.pt $LF || { grep -q OutOfMemoryError $L/${DS}_${U}_field.log 2>/dev/null && \
             step ${DS}_${U}_field_cpu $M/${U}_r1_lf64.pt $LF --cpu-opt-state; } || return 1;;
      esac
      [ -z "${DRY:-}" ] && [ -e $M/${U}_r1.pt ] && [ -e $M/${U}_r1_lf64.pt ] && rm -f $FIT       # the candidate is no longer needed
    done
    log "$DS maps done"
  done
}
encoders() {  # optional: encoders on the rebuilt maps, into ckpts/v2_rebuild (ckpts/v2 untouched)
  local DS E SC U
  for DS in 7scenes 12scenes; do cfg $DS; mkdir -p ckpts/v2_rebuild/$DS
    for E in $PICKS; do SC=${E%:*}; U=$(u $SC)
      step ${DS}_${U}_enc ckpts/v2_rebuild/$DS/${U}_encoders.pt python scripts/train_encoders.py --root $ROOT --dataset $DS \
        --scene $SC --sfm-pose-dir $PGT --map $M/${U}_r1.pt --head-ckpt $K/head_sel.pt --out ckpts/v2_rebuild/$DS/${U}_encoders.pt || return 1
    done
  done
}
eval_runs() {  # Table I (7Scenes, seeds 0-2) and Table II (12Scenes, seed 0) configuration, into new result folders
  local DS E SC U SEED OUT
  for DS in 7scenes 12scenes; do cfg $DS
    for SEED in $([ $DS = 7scenes ] && echo 0 1 2 || echo 0); do OUT=results/rebuild_$(basename $ENC)_${DS}_seed$SEED
      for E in $PICKS; do SC=${E%:*}; U=$(u $SC)
        step ${DS}_${U}_run$SEED $OUT/${DS}_${U}.json python scripts/run_7scenes.py --root $ROOT --dataset $DS --scene $SC \
          --sfm-pose-dir $PGT --map $M/${U}_r1.pt --corr-feats $M/${U}_r1_lf64.pt --encoders $ENC/$DS/${U}_encoders.pt \
          --queries 100 --device cuda --top-k 3 --iters 60 --temperature-split 0.5 --temp-calib-frames 30 --anchor-stride 20 \
          --submap-radius 3.0 --renderer gsplat --width 640 --height 480 --fuse-rule yager --lambda-u 2.0 --shortlist 10 \
          --opinion-mode edl --corr-refine-iters 60 --correction --support-power 1.0 --corr-conf-min 0.2 --gamma 3.0 \
          --seed $SEED --oom-retries 40 --corr-rounds 8 --corr-matcher stage --corr-stage-warm mast3r --corr-feat-stage 3 \
          --corr-stop-dt 0.0002 --corr-stop-dr 0.01 --out $OUT
      done
    done
  done
}

case ${1:-} in
  check) check ;;
  data) data ;;
  check-data) check_data ;;
  maps) maps ;;
  encoders) encoders ;;
  eval) eval_runs ;;
  all) check && data && maps && log "ALL DONE" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
