"""Unified config loading, CLI overrides, validation, and path helpers."""
from __future__ import annotations
import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any
try:
    import yaml
except ModuleNotFoundError:

    class _MiniYAML:

        @staticmethod
        def safe_load(stream) -> Any:
            text = stream.read() if hasattr(stream, 'read') else str(stream)
            if not text.strip():
                return None
            stripped = text.lstrip()
            if stripped.startswith('{') or stripped.startswith('['):
                return json.loads(text)
            root: dict[str, Any] = {}
            stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
            for raw_line in text.splitlines():
                content = raw_line.split('#', 1)[0].rstrip()
                if not content.strip():
                    continue
                indent = len(content) - len(content.lstrip(' '))
                line = content.strip()
                if ':' not in line:
                    raise ValueError(f'unsupported YAML line without key separator: {raw_line!r}')
                key, value = line.split(':', 1)
                key = key.strip().strip('\'"')
                value = value.strip()
                while indent <= stack[-1][0]:
                    stack.pop()
                parent = stack[-1][1]
                if value == '':
                    child: dict[str, Any] = {}
                    parent[key] = child
                    stack.append((indent, child))
                else:
                    parent[key] = _MiniYAML._parse_scalar(value)
            return root

        @staticmethod
        def _parse_scalar(value: str) -> Any:
            if value.startswith('{') and value.endswith('}'):
                inner = value[1:-1].strip()
                if not inner:
                    return {}
                parsed: dict[str, Any] = {}
                for part in inner.split(','):
                    if ':' not in part:
                        raise ValueError(f'unsupported inline mapping entry: {part!r}')
                    key, item_value = part.split(':', 1)
                    parsed[key.strip().strip('\'"')] = _MiniYAML._parse_scalar(item_value.strip())
                return parsed
            if value.startswith('[') and value.endswith(']'):
                inner = value[1:-1].strip()
                if not inner:
                    return []
                return [_MiniYAML._parse_scalar(part.strip()) for part in inner.split(',')]
            if value.startswith('"') and value.endswith('"') or (value.startswith("'") and value.endswith("'")):
                return value[1:-1]
            lower = value.lower()
            if lower in {'null', 'none'}:
                return None
            if lower == 'true':
                return True
            if lower == 'false':
                return False
            try:
                return int(value)
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                return value
    yaml = _MiniYAML()

class DotDict:
    """Read-only dot-access wrapper around nested dictionaries."""

    def __init__(self, data: dict[str, Any]):
        object.__setattr__(self, '_data', {})
        for key, value in data.items():
            self._data[key] = DotDict(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(f'config field {name!r} does not exist') from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name == '_data':
            object.__setattr__(self, name, value)
            return
        if name.startswith('_'):
            self._data[name] = value
            return
        raise AttributeError(f'config is frozen, cannot assign {name!r}; use CLI overrides or load_and_merge(overrides=...)')

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __repr__(self) -> str:
        return f'DotDict({self.to_dict()})'

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self._data.items():
            out[key] = value.to_dict() if isinstance(value, DotDict) else value
        return out

def raw_compat_is_enabled(config: DotDict) -> bool:
    """Return whether training or inference needs the raw compatibility scorer."""
    loss = getattr(config, 'loss', None)
    if loss is not None and float(getattr(loss, 'raw_aux_weight', 0.0) or 0.0) > 0.0:
        return True
    return raw_compat_inference_is_enabled(config)

def raw_compat_inference_is_enabled(config: DotDict) -> bool:
    """Return whether configured inference or its validation choices need raw items."""
    rrf = getattr(config, 'rrf', None)
    if rrf is None:
        return False
    if float(getattr(rrf, 'raw_compat_weight', 0.0) or 0.0) > 0.0:
        return True
    param_choices = getattr(rrf, 'param_choices', None)
    choices = getattr(param_choices, 'raw_compat_weight', []) if param_choices is not None else []
    return any((float(value) > 0.0 for value in choices))

def parse_args_and_load(description: str) -> DotDict:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    known, unknown = parser.parse_known_args()
    overrides = _parse_overrides(unknown)
    return load_and_merge(known.config, overrides)

def load_from_run_or_config(config_path: str | None, run_reference_path: str | os.PathLike[str] | None, override_args: list[str] | None=None, fallback_config_path: str | None=None) -> DotDict:
    inferred_overrides = infer_run_overrides(run_reference_path)
    cli_overrides = _parse_overrides(list(override_args or []))
    inferred_overrides.update(cli_overrides)
    resolved_config_path = resolve_config_path(config_path, run_reference_path, fallback_config_path)
    return load_and_merge(resolved_config_path, inferred_overrides)

def load_from_checkpoint_or_config(config_path: str | None, checkpoint_path: str | os.PathLike[str] | None, override_args: list[str] | None=None, fallback_config_path: str | None=None) -> DotDict:
    return load_from_run_or_config(config_path, checkpoint_path, override_args, fallback_config_path)

def load_and_merge(yaml_path: str, overrides: dict[str, Any] | None=None) -> DotDict:
    with open(yaml_path, 'r', encoding='utf-8') as file:
        raw = yaml.safe_load(file)
    if raw is None:
        raise ValueError(f'YAML file is empty or invalid: {yaml_path}')
    if overrides:
        for dotted_key, value in overrides.items():
            _set_nested(raw, dotted_key, value)
    _normalize_rrf_tune_every(raw)
    _normalize_mosid_ablation_defaults(raw)
    _normalize_mosid_beam_sizes(raw)
    _reject_removed_options(raw)
    _validate_types(raw)
    _validate_train(raw)
    _validate_inference(raw)
    _validate_rqvae(raw)
    _validate_mosid(raw)
    _check_required(raw)
    _build_paths(raw)
    return DotDict(raw)

def _normalize_rrf_tune_every(raw: dict[str, Any]) -> None:
    rrf = raw.get('rrf')
    if isinstance(rrf, dict) and 'tune_every' not in rrf and ('eval_every' in rrf):
        rrf['tune_every'] = rrf['eval_every']

def _normalize_mosid_ablation_defaults(raw: dict[str, Any]) -> None:
    if not _looks_like_mosid_config(raw):
        return
    loss = raw.get('loss')
    if isinstance(loss, dict):
        loss.setdefault('raw_aux_weight', 0.0)
        loss.setdefault('raw_temperature', 0.07)
    rrf = raw.get('rrf')
    if isinstance(rrf, dict):
        rrf.setdefault('gs_on_valid', False)
        rrf.setdefault('rank_fusion_weight', 1.0)
        rrf.setdefault('raw_compat_weight', 1.0)

def _normalize_mosid_beam_sizes(raw: dict[str, Any]) -> None:
    """Map the legacy MoSID beam size to both split-specific settings."""
    if not _looks_like_mosid_config(raw):
        return
    inference = raw.get('inference')
    if not isinstance(inference, dict) or 'beam_size' not in inference:
        return
    legacy_beam_size = inference.pop('beam_size')
    inference.setdefault('valid_beam_size', legacy_beam_size)
    inference.setdefault('test_beam_size', legacy_beam_size)

def _parse_overrides(args: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith('--'):
            raise ValueError(f'cannot parse CLI argument {token!r}; expected --key.subkey value')
        key = token[2:]
        if index + 1 >= len(args) or args[index + 1].startswith('--'):
            overrides[key] = True
            index += 1
            continue
        overrides[key] = _infer_type(args[index + 1])
        index += 2
    return overrides

def _infer_type(value_str: str) -> Any:
    stripped = value_str.strip()
    if stripped.startswith('[') and stripped.endswith(']'):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [_infer_type(item.strip().strip('\'"')) for item in inner.split(',')]
    lower = value_str.lower()
    if lower in {'none', 'null'}:
        return None
    if lower == 'true':
        return True
    if lower == 'false':
        return False
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str

def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split('.')
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
_REMOVED_OPTIONS = {'model.lora_dropout': 'LoRA dropout has been removed; projection LoRA now uses a fixed dropout value of 0.0.', 'negative_sampling': 'Explicit negative sampling has been removed; item loss now always uses in-batch negatives.', 'rrf.search_grid': 'RRF tuning compares the current setting with at most rrf.sample_count random rrf.param_choices alternatives per tune epoch.', 'train.log_every': 'Batch-level train logging has been removed because it disrupts the visual progress bar.'}
_RQVAE_REMOVED_OPTIONS = {'model.mlp_style': 'RQ-VAE now always uses the LETTER-style MLP.', 'model.forward_mode': 'RQ-VAE now always uses the LETTER-style STE quantization path.', 'model.codebook_normalize': 'RQ-VAE codebook normalization has been removed.', 'model.dead_code_threshold': 'RQ-VAE batch dead-code reset has been removed.', 'model.dead_code_refresh_every': 'RQ-VAE low-frequency code refresh has been removed.', 'model.dead_code_min_count': 'RQ-VAE low-frequency code refresh has been removed.', 'gumbel': 'RQ-VAE Gumbel-Softmax quantization has been removed.', 'train.optimizer': 'RQ-VAE training now always uses torch.optim.AdamW in train_rqvae.py.'}
_MOSID_REMOVED_OPTIONS = {'model.train_objective': 'MoSID Stage 2 now always uses last-target supervision; remove model.train_objective.', 'model.t5_path': 'MoSID Stage 2 uses a lightweight encoder-decoder SID model and does not load T5.', 'data.drop_invalid_targets': 'MoSID P2/P8 keeps every target item and masks unavailable content channels in L_rg.', 'model.enable_item_intent': 'Item-level intent bottleneck has been removed from the maintained P2/P8 code path.', 'model.enable_cross_attention': 'SID-token cross-attention has been removed from the maintained P2/P8 code path.', 'model.enable_reliability_gate': 'Reliability gate has been removed from the maintained P2/P8 code path.', 'model.enable_relation_bias': 'Relation bias has been removed from the maintained P2/P8 code path.', 'model.train_sid_embedding': 'MoSID Stage 2 now always freezes Stage 1 SID codebook embeddings.', 'rrf.gamma': 'Rank-Space Unified RRF has removed the redundant branch-overlap bonus.', 'rrf.param_choices.gamma': 'Rank-Space Unified RRF has removed the redundant branch-overlap bonus.', 'rrf.test_sample_count': 'Use rrf.gs_on_valid to enable exhaustive valid-set RRF grid search before test.', 'model.tie_output_heads': 'MoSID Stage 2 always uses independent layer-wise output heads.', 'model.mhrab': 'MHRAB has been removed after experiments showed it did not improve the model.', 'model.msmoe': 'MSMoE has been removed after experiments showed it did not improve the model.', 'model.dynamic_token': 'Dynamic missing token has been removed; MoSID now uses observed-only SID artifacts.', 'model.dynamic_token_mode': 'Dynamic missing token has been removed; remove model.dynamic_token_mode from the config.', 'model.history_attention_mode': 'History attention mode has been fixed to causal; remove model.history_attention_mode from the config.', 'model.shared_history_encoder': 'Shared history encoder has been removed; MoSID now always uses one encoder per active channel.', 'train.teacher_forced_eval': 'MoSID validation now always uses beam ranking; remove train.teacher_forced_eval from the config.', 'loss.lambda_item': 'Item-level recommendation loss has been removed from the maintained P2/P8 code path.', 'loss.lambda_rel': 'Reliability consistency loss has been removed from the maintained P2/P8 code path.', 'loss.lambda_miss': 'Candidate-side Virtual SID loss has been removed from the maintained P2/P8 code path.', 'loss.lambda_align': 'Virtual SID alignment loss has been removed from the maintained P2/P8 code path.', 'loss.lambda_rel_cons': 'Reliability consistency loss has been removed from the maintained P2/P8 code path.', 'loss.lambda_recover': 'Legacy recovery loss has been removed from the maintained P2/P8 code path.', 'loss.tau_item': 'Item-level intent loss has been removed from the maintained P2/P8 code path.', 'loss.tau_D': 'Reliability consistency loss has been removed from the maintained P2/P8 code path.', 'loss.lambda_collision': 'Collision-aware reliability loss has been removed from the maintained P2/P8 code path.', 'missing': 'The missing recovery module and dynamic missing token path have been removed.', 'recover': 'Legacy recovery has been removed from the maintained P2/P8 code path.', 'virtual': 'Virtual SID and history refinement have been removed from the maintained P2/P8 code path.'}

def _reject_removed_options(data: dict[str, Any]) -> None:
    for dotted_key, message in _REMOVED_OPTIONS.items():
        _reject_removed_option(data, dotted_key, message)
    if _looks_like_rqvae_config(data):
        for dotted_key, message in _RQVAE_REMOVED_OPTIONS.items():
            _reject_removed_option(data, dotted_key, message)
    if _looks_like_mosid_config(data):
        for dotted_key, message in _MOSID_REMOVED_OPTIONS.items():
            _reject_removed_option(data, dotted_key, message)

def _reject_removed_option(data: dict[str, Any], dotted_key: str, message: str) -> None:
    current = data
    keys = dotted_key.split('.')
    for key in keys[:-1]:
        if not isinstance(current, dict) or key not in current:
            break
        current = current[key]
    else:
        if isinstance(current, dict) and keys[-1] in current:
            raise ValueError(f'removed config option {dotted_key!r}: {message}')

def _looks_like_rqvae_config(data: dict[str, Any]) -> bool:
    model = data.get('model', {})
    return isinstance(model, dict) and 'commitment_weight' in model and ('quant_loss_weight' in model)
_RQVAE_KMEANS_CENTER_INIT = {'global_mean', 'full_exist'}
_RQVAE_ENC_DEC_TYPES = {'mpl': 'MPL', 'linear': 'Linear'}
_RQVAE_UNIFIED_ENC_DEC_MODES = {'linear', 'mlp'}

def _validate_rqvae(raw: dict[str, Any]) -> None:
    if not _looks_like_rqvae_config(raw):
        return
    enc_dec_type = _get_nested(raw, 'model.enc_dec_type')
    enc_dec_type_key = str(enc_dec_type or 'MPL').lower()
    if enc_dec_type_key not in _RQVAE_ENC_DEC_TYPES:
        raise ValueError(f'model.enc_dec_type must be one of {sorted(_RQVAE_ENC_DEC_TYPES.values())!r}, got {enc_dec_type!r}')
    _set_nested(raw, 'model.enc_dec_type', _RQVAE_ENC_DEC_TYPES[enc_dec_type_key])
    enc_dec_mode = _get_nested(raw, 'model.enc_dec_mode')
    if enc_dec_mode is not None:
        enc_dec_mode = str(enc_dec_mode).lower()
        if enc_dec_mode not in _RQVAE_UNIFIED_ENC_DEC_MODES:
            raise ValueError(f'model.enc_dec_mode must be one of {sorted(_RQVAE_UNIFIED_ENC_DEC_MODES)!r}, got {enc_dec_mode!r}')
        _set_nested(raw, 'model.enc_dec_mode', enc_dec_mode)
    kmeans_center_init = _get_nested(raw, 'model.kmeans_center_init')
    if kmeans_center_init is not None:
        kmeans_center_init = str(kmeans_center_init).lower()
        if kmeans_center_init not in _RQVAE_KMEANS_CENTER_INIT:
            raise ValueError(f'model.kmeans_center_init must be one of {sorted(_RQVAE_KMEANS_CENTER_INIT)!r}, got {kmeans_center_init!r}')
        _set_nested(raw, 'model.kmeans_center_init', kmeans_center_init)
_TYPE_RULES: dict[str, type] = {'experiment.stage': str, 'experiment.domain_scope': str, 'experiment.protocol_version': str, 'train.lr': float, 'train.lr_rec': float, 'train.lr_id': float, 'train.weight_decay': float, 'train.optimizer': str, 'train.learner': str, 'train.batch_size': int, 'train.grad_accum_steps': int, 'train.max_epochs': int, 'train.eval_every': int, 'train.patience': int, 'train.warmup_steps': int, 'train.lr_scheduler_type': str, 'train.es_warmup': int, 'train.es_metric': str, 'train.semantic_collision_rate_max': float, 'train.semantic_max_leaf': int, 'train.prefix_semantic_weights': list, 'train.prefix_semantic_tolerance': float, 'train.early_stop_metric': str, 'train.early_stop_min_delta': float, 'train.grad_clip': float, 'train.num_workers': int, 'train.gpu_id': int, 'train.seed': int, 'train.tensorboard': bool, 'train.run_dir_file': str, 'train.progress_bar': bool, 'model.architecture': str, 'model.hidden_units': int, 'model.num_blocks': int, 'model.num_heads': int, 'model.ff_dim': int, 'model.dropout_rate': float, 'model.attn_dropout_prob': float, 'model.hidden_act': str, 'model.initializer_range': float, 'contrast.similarity': str, 'contrast.temperature': float, 'contrast.lambda': float, 'contrast.fp32_similarity': bool, 'model.train_objective': str, 'model.train_stage': str, 'model.report_scope': str, 'model.candidate_policy': str, 'model.eval_mode': str, 'model.score_fusion': str, 'model.variant_selection': str, 'model.use_id_embedding': bool, 'model.apply_temperature_to_ce': bool, 'model.plm_source': str, 'model.plm_fusion': str, 'model.plm_size': int, 'model.hidden_size': int, 'model.inner_size': int, 'model.adaptor_dropout_prob': float, 'model.adaptor_layers': list, 'model.temperature': float, 'model.n_exps': int, 'model.freeze_plm_embedding': bool, 'model.use_moe_noise': bool, 'model.norm_first': bool, 'model.modal_type': str, 'model.item_mm_fusion': str, 'model.seq_mm_fusion': str, 'model.discovery_method': str, 'model.missing_policy': str, 'model.zero_embedding_adaptor_bias': str, 'model.text_emb_size': int, 'model.vision_emb_size': int, 'model.discovery_every': int, 'model.freeze_content_embedding': bool, 'model.layer_norm_eps': float, 'model.embed_dim': int, 'model.codebook_size': int, 'model.n_layers': int, 'model.code_dim': int, 'model.baseline_name': str, 'model.implementation': str, 'model.variant': str, 'model.backbone': str, 'model.sid_source': str, 'model.pretrained_t5': bool, 'model.semantic_embed_dim': int, 'model.transformer_dim': int, 'model.relative_attention_num_buckets': int, 'model.relative_attention_max_distance': int, 'model.enc_dec_type': str, 'model.enc_dec_mode': str, 'model.hidden_dims': list, 'model.auxiliary_losses': list, 'model.dropout_prob': float, 'model.use_batch_norm': bool, 'model.activation': str, 'model.reconstruction_loss': str, 'model.vq_type': str, 'model.distance': str, 'model.use_linear_projection': bool, 'model.kmeans_init': bool, 'model.kmeans_iters': int, 'model.latent_block_dim': int, 'model.use_bpd': bool, 'model.encoder_layers': int, 'model.decoder_layers': int, 'model.dropout': float, 'model.random_init': bool, 'model.d_model': int, 'model.d_kv': int, 'model.num_encoder_layers': int, 'model.num_decoder_layers': int, 'model.d_ff': int, 'model.e_dim': int, 'model.input_dim': int, 'model.tie_word_embeddings': bool, 'model.beta': float, 'model.code_length': int, 'model.activation_function': str, 'model.commitment_weight': float, 'model.quant_loss_weight': float, 'model.assignment_mode': str, 'model.sk_epsilons': list, 'model.sk_iters': int, 'model.use_sinkhorn_train': bool, 'model.use_sinkhorn_eval': bool, 'model.diversity_loss_weight': float, 'model.diversity_num_groups': int, 'model.codebook_group_clusters': int, 'model.codebook_group_refresh_every': int, 'model.cf_loss_weight': float, 'model.cf_loss_source': str, 'model.cf_loss_modalities': list, 'model.cf_loss_temperature': float, 'model.cross_modal_pred_weight': float, 'model.kmeans_center_init': str, 'model.kmeans_n_init': int, 'model.kmeans_init_max_iter': int, 'data.sm_prob': float, 'data.max_len': int, 'data.modality': str, 'data.text_embedding_file': str, 'data.vision_embedding_file': str, 'data.use_missing': bool, 'model.t5_path': str, 'model.sid_artifact_mode': str, 'model.encoder_mode': str, 'model.item_present': str, 'model.sid_embed_dim': int, 'model.active_channels': list, 'model.hidden_dim': int, 'model.n_heads': int, 'model.history_layers': int, 'model.ff_dim': int, 'model.ff_activation': str, 'model.channels': list, 'loss.branch_weights': dict, 'loss.raw_aux_weight': float, 'loss.raw_temperature': float, 'loss.raw_use_history_reliability': bool, 'loss.raw_prefix_hard_negative_counts_by_lcp': list, 'loss.raw_random_negatives': bool, 'loss.memory_orth_weight': float, 'loss.memory_orth_direct': bool, 'loss.memory_orth_center': bool, 'loss.memory_orth_uniform_weights': bool, 'loss.loss_type': str, 'loss.temperature': float, 'loss.code_loss_weight': float, 'loss.recon_loss_weight': float, 'loss.vq_loss_weight': float, 'loss.qs_loss_weight': float, 'loss.balance_loss_weight': float, 'loss.type': str, 'loss.auxiliary_losses': list, 'sid.depth': int, 'sid.codebook_size': int, 'sid.code_dim': int, 'sid.artifact_code_range': list, 'sid.padding_code': int, 'vocab.type': str, 'vocab.use_sentencepiece': bool, 'vocab.namespace_by_modality_and_depth': bool, 'vocab.pad_token_id': int, 'vocab.bos_token_id': int, 'vocab.eos_token_id': int, 'vocab.decoder_start_token_id': int, 'tasks.names': list, 'tasks.prompt_tokens_per_task': int, 'tasks.scheduler': str, 'tasks.equal_sampling': bool, 'tasks.sequence_source': str, 'tasks.expand_sequence_prefixes': bool, 'tasks.qla_catalog': str, 'tasks.use_valid_test_interaction_labels': bool, 'tasks.epoch_micro_batch_budget': str, 'precision.amp': bool, 'precision.dtype': str, 'inference.beam_size': int, 'inference.valid_beam_size': int, 'inference.test_beam_size': int, 'inference.beam_logits_chunk_size': int, 'inference.top_k': list, 'rrf.k': float, 'rrf.rank_fusion_weight': float, 'rrf.raw_compat_weight': float, 'rrf.sample_count': int, 'rrf.gs_on_valid': bool, 'rrf.param_choices': dict, 'rrf.selection_metric': str, 'rrf.tune_every': int, 'eval.batch_size': int, 'eval.top_k': list, 'eval.max_batches': int, 'evaluation.batch_size': int, 'evaluation.top_k': list, 'evaluation.max_batches': int, 'evaluation.selection_metric': str, 'reranking.strategy': str, 'reranking.both_branch_divisor': float, 'reranking.both_branch_bonus': float, 'reranking.tunable_weights': bool, 'artifacts.contract_version': int, 'artifacts.text_sid_path': str, 'artifacts.text_metadata_path': str, 'artifacts.vision_sid_path': str, 'artifacts.vision_metadata_path': str, 'artifacts.require_hash_match': bool, 'artifacts.allow_shared_sid_fallback': bool, 'pretrain.enabled': bool, 'pretrain.max_epochs': int, 'pretrain.batch_size': int, 'pretrain.grad_accum_steps': int, 'pretrain.optimizer': str, 'pretrain.lr': float, 'pretrain.weight_decay': float, 'pretrain.eval_every': int, 'pretrain.patience': int, 'pretrain.output_dir': str, 'pretrain.gpu_id': int, 'pretrain.seed': int, 'pretrain.tensorboard': bool, 'pretrain.mask_ratio': float, 'pretrain.mip_weight': float, 'pretrain.sp_weight': float, 'pretrain.nip_weight': float, 'pretrain.cmcl_weight': float, 'pretrain.use_projection': bool, 'pretrain.seq_drop_prob': float, 'pretrain.mixup_max_prob': float, 'frqud.enabled': bool, 'frqud.adaptive_selection': bool, 'frqud.hot_threshold_ratio': float, 'frqud.usage_momentum': float, 'frqud.gumbel_tau': float, 'frqud.gumbel_hard_switch_epoch': int, 'frqud.use_indicator_ste': bool}

def _validate_types(raw: dict[str, Any]) -> None:
    for dotted_key, expected_type in _TYPE_RULES.items():
        value = _get_nested(raw, dotted_key)
        if value is None:
            continue
        if expected_type in {int, float} and isinstance(value, bool):
            raise TypeError(f'config type mismatch for {dotted_key!r}: expected {expected_type.__name__}, got bool ({value!r})')
        if expected_type is int and isinstance(value, float) and (value == int(value)):
            _set_nested(raw, dotted_key, int(value))
            continue
        if expected_type is float and isinstance(value, int):
            _set_nested(raw, dotted_key, float(value))
            continue
        if expected_type is float and isinstance(value, str):
            try:
                _set_nested(raw, dotted_key, float(value))
                continue
            except ValueError:
                pass
        if not isinstance(value, expected_type):
            raise TypeError(f'config type mismatch for {dotted_key!r}: expected {expected_type.__name__}, got {type(value).__name__} ({value!r})')

def _validate_train(raw: dict[str, Any]) -> None:
    train = raw.get('train', {})
    if not isinstance(train, dict):
        return
    for key in ('batch_size', 'grad_accum_steps'):
        value = train.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f'train.{key} must be positive, got {value}')
    es_warmup = train.get('es_warmup')
    if es_warmup is not None and int(es_warmup) < 0:
        raise ValueError(f'train.es_warmup must be non-negative, got {es_warmup}')
    es_metric = train.get('es_metric')
    allowed_es_metrics = {'collision', 'prefix_semantic'}
    if es_metric is not None and str(es_metric).lower() not in allowed_es_metrics:
        raise ValueError(f'train.es_metric must be one of {sorted(allowed_es_metrics)!r}, got {es_metric!r}')
    collision_max = train.get('semantic_collision_rate_max')
    if collision_max is not None and (not 0.0 <= float(collision_max) <= 1.0):
        raise ValueError(f'train.semantic_collision_rate_max must be in [0, 1], got {collision_max}')
    max_leaf = train.get('semantic_max_leaf')
    if max_leaf is not None and int(max_leaf) <= 0:
        raise ValueError(f'train.semantic_max_leaf must be positive, got {max_leaf}')
    prefix_weights = train.get('prefix_semantic_weights')
    if prefix_weights is not None and (not prefix_weights or any((float(weight) <= 0.0 for weight in prefix_weights))):
        raise ValueError('train.prefix_semantic_weights must contain positive values')
    semantic_tolerance = train.get('prefix_semantic_tolerance')
    if semantic_tolerance is not None and float(semantic_tolerance) < 0.0:
        raise ValueError(f'train.prefix_semantic_tolerance must be non-negative, got {semantic_tolerance}')

def _validate_inference(raw: dict[str, Any]) -> None:
    inference = raw.get('inference', {})
    if not isinstance(inference, dict):
        return
    for field in ('beam_size', 'valid_beam_size', 'test_beam_size'):
        beam_size = inference.get(field)
        if beam_size is not None and int(beam_size) <= 0:
            raise ValueError(f'inference.{field} must be positive, got {beam_size}')
    chunk_size = inference.get('beam_logits_chunk_size')
    if chunk_size is not None and int(chunk_size) < 0:
        raise ValueError(f'inference.beam_logits_chunk_size must be non-negative, got {chunk_size}')
_MOSID_REQUIRED_FIELDS = ['data.base_path', 'data.dataset', 'data.mode', 'data.sm_prob', 'data.max_len', 'model.channels', 'model.active_channels', 'model.sid_embed_dim', 'model.hidden_dim', 'model.n_heads', 'model.history_layers', 'model.decoder_layers', 'model.ff_dim', 'model.dropout', 'loss.branch_weights', 'loss.memory_orth_weight', 'inference.valid_beam_size', 'inference.test_beam_size', 'evaluation.batch_size', 'evaluation.top_k']
_MOSID_RRF_REQUIRED_FIELDS = ['rrf.k', 'rrf.sample_count', 'rrf.gs_on_valid', 'rrf.param_choices', 'rrf.selection_metric', 'rrf.tune_every']
_MOSID_SID_ARTIFACT_MODES = {'separate', 'unified'}
_MOSID_ENCODER_MODES = {'causal', 'bidirection'}
_MOSID_FF_ACTIVATIONS = {'gelu', 'leaky_relu', 'relu', 'swiglu'}
_MOSID_UNIFIED_BLOCK_VIEW_CHANNELS = {'text', 'vision', 'collab'}
_MOSID_MEMORY_ORTH_CHANNELS = {'text', 'vision', 'collab'}

def _looks_like_mosid_config(data: dict[str, Any]) -> bool:
    model = data.get('model', {})
    return isinstance(model, dict) and ('t5_path' in model or 'sid_embed_dim' in model or 'active_channels' in model or (model.get('channels') == ['text', 'vision', 'collab'])) or 'rrf' in data or 'virtual' in data

def _validate_mosid(raw: dict[str, Any]) -> None:
    if not _looks_like_mosid_config(raw):
        return
    for dotted_key in _MOSID_REQUIRED_FIELDS:
        if not _has_nested(raw, dotted_key):
            raise ValueError(f'missing required config field {dotted_key!r}')
        if _get_nested(raw, dotted_key) is None:
            raise ValueError(f'missing required config field {dotted_key!r}')
    mode = _get_nested(raw, 'data.mode')
    if mode not in {'nm', 'sm'}:
        raise ValueError(f"data.mode must be 'nm' or 'sm', got {mode!r}")
    max_len = _get_nested(raw, 'data.max_len')
    if int(max_len) <= 0:
        raise ValueError(f'data.max_len must be positive, got {max_len}')
    sm_prob = float(_get_nested(raw, 'data.sm_prob'))
    if sm_prob < 0.0 or sm_prob > 1.0:
        raise ValueError(f'data.sm_prob must be in [0, 1], got {sm_prob}')
    channels = [str(channel) for channel in _get_nested(raw, 'model.channels')]
    if not channels or len(set(channels)) != len(channels):
        raise ValueError(f'model.channels must be a non-empty unique channel list, got {channels!r}')
    active_channels = [str(channel) for channel in _get_nested(raw, 'model.active_channels')]
    if not active_channels or len(set(active_channels)) != len(active_channels):
        raise ValueError(f'model.active_channels must be a non-empty unique channel list, got {active_channels!r}')
    unknown_active_channels = [channel for channel in active_channels if channel not in channels]
    if unknown_active_channels:
        raise ValueError(f'model.active_channels contains unknown channels: {unknown_active_channels!r}')
    sid_artifact_mode = str(_get_nested(raw, 'model.sid_artifact_mode') or 'separate').lower()
    if sid_artifact_mode not in _MOSID_SID_ARTIFACT_MODES:
        raise ValueError(f'model.sid_artifact_mode must be one of {sorted(_MOSID_SID_ARTIFACT_MODES)!r}, got {sid_artifact_mode!r}')
    if sid_artifact_mode == 'unified' and set(active_channels) != _MOSID_UNIFIED_BLOCK_VIEW_CHANNELS:
        raise ValueError(f"model.sid_artifact_mode='unified' requires active_channels {sorted(_MOSID_UNIFIED_BLOCK_VIEW_CHANNELS)!r}, got {active_channels!r}")
    encoder_mode = str(_get_nested(raw, 'model.encoder_mode') or 'causal').lower()
    if encoder_mode not in _MOSID_ENCODER_MODES:
        raise ValueError(f'model.encoder_mode must be one of {sorted(_MOSID_ENCODER_MODES)!r}, got {encoder_mode!r}')
    _set_nested(raw, 'model.encoder_mode', encoder_mode)
    ff_activation = str(_get_nested(raw, 'model.ff_activation') or 'relu').lower()
    if ff_activation not in _MOSID_FF_ACTIVATIONS:
        raise ValueError(f'model.ff_activation must be one of {sorted(_MOSID_FF_ACTIVATIONS)!r}, got {ff_activation!r}')
    _set_nested(raw, 'model.ff_activation', ff_activation)
    memory_orth_weight = float(_get_nested(raw, 'loss.memory_orth_weight'))
    if memory_orth_weight < 0.0:
        raise ValueError(f'loss.memory_orth_weight must be non-negative, got {memory_orth_weight}')
    if memory_orth_weight > 0.0 and set(active_channels) != _MOSID_MEMORY_ORTH_CHANNELS:
        raise ValueError(f'loss.memory_orth_weight > 0 requires active_channels {sorted(_MOSID_MEMORY_ORTH_CHANNELS)!r}, got {active_channels!r}')
    is_single_channel = len(active_channels) == 1
    if is_single_channel:
        if 'rrf' in raw:
            raise ValueError('single-channel MoSID config must not define rrf; use direct single-trie beam ranking')
        merge_branches = _get_nested(raw, 'inference.merge_branches')
        if merge_branches is not None:
            raise ValueError('single-channel MoSID config must not define inference.merge_branches')
    else:
        for dotted_key in _MOSID_RRF_REQUIRED_FIELDS:
            if not _has_nested(raw, dotted_key):
                raise ValueError(f'missing required config field {dotted_key!r}')
            if _get_nested(raw, dotted_key) is None:
                raise ValueError(f'missing required config field {dotted_key!r}')
    sid_embed_dim = _get_nested(raw, 'model.sid_embed_dim')
    if int(sid_embed_dim) <= 0:
        raise ValueError(f'model.sid_embed_dim must be positive, got {sid_embed_dim}')
    for dotted_key in ('model.hidden_dim', 'model.n_heads', 'model.history_layers', 'model.decoder_layers', 'model.ff_dim'):
        value = int(_get_nested(raw, dotted_key))
        if value <= 0:
            raise ValueError(f'{dotted_key} must be positive, got {value}')
    hidden_dim = int(_get_nested(raw, 'model.hidden_dim'))
    n_heads = int(_get_nested(raw, 'model.n_heads'))
    if hidden_dim % n_heads != 0:
        raise ValueError(f'model.hidden_dim must be divisible by model.n_heads, got {hidden_dim} and {n_heads}')
    dropout = float(_get_nested(raw, 'model.dropout'))
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError(f'model.dropout must be in [0, 1), got {dropout}')
    branch_weights = _get_nested(raw, 'loss.branch_weights')
    if set(branch_weights) != set(channels):
        raise ValueError(f'loss.branch_weights must define exactly {channels!r}, got {sorted(branch_weights)!r}')
    if any((float(weight) <= 0.0 for weight in branch_weights.values())):
        raise ValueError(f'loss.branch_weights values must be positive, got {branch_weights!r}')
    raw_aux_weight = float(_get_nested(raw, 'loss.raw_aux_weight'))
    if raw_aux_weight < 0.0:
        raise ValueError(f'loss.raw_aux_weight must be non-negative, got {raw_aux_weight}')
    raw_temperature = float(_get_nested(raw, 'loss.raw_temperature'))
    if raw_temperature <= 0.0:
        raise ValueError(f'loss.raw_temperature must be positive, got {raw_temperature}')
    for dotted_key in ('inference.valid_beam_size', 'inference.test_beam_size'):
        beam_size = int(_get_nested(raw, dotted_key))
        if beam_size <= 0:
            raise ValueError(f'{dotted_key} must be positive, got {beam_size}')
    top_k = _get_nested(raw, 'evaluation.top_k')
    if not top_k or any((int(k) <= 0 for k in top_k)):
        raise ValueError(f'evaluation.top_k must contain positive integers, got {top_k!r}')
    if not is_single_channel:
        for dotted_key in ('rrf.rank_fusion_weight', 'rrf.raw_compat_weight'):
            value = float(_get_nested(raw, dotted_key))
            if value < 0.0:
                raise ValueError(f'{dotted_key} must be non-negative, got {value}')
        if float(_get_nested(raw, 'rrf.rank_fusion_weight')) == 0.0 and float(_get_nested(raw, 'rrf.raw_compat_weight')) == 0.0:
            raise ValueError('rrf.rank_fusion_weight and rrf.raw_compat_weight must not both be zero')
        rrf_k = float(_get_nested(raw, 'rrf.k'))
        if rrf_k <= 0.0:
            raise ValueError(f'rrf.k must be positive, got {rrf_k}')
        selection_metric = _get_nested(raw, 'rrf.selection_metric')
        if not selection_metric:
            raise ValueError('rrf.selection_metric must be non-empty')
        sample_count = _get_nested(raw, 'rrf.sample_count')
        if sample_count is not None and int(sample_count) < 0:
            raise ValueError(f'rrf.sample_count must be non-negative, got {sample_count}')
        rrf_tune_every = _get_nested(raw, 'rrf.tune_every')
        if rrf_tune_every is not None and int(rrf_tune_every) <= 0:
            raise ValueError(f'rrf.tune_every must be positive, got {rrf_tune_every}')
        param_choices = _get_nested(raw, 'rrf.param_choices')
        if param_choices is not None:
            _validate_rrf_param_choices(param_choices)
    if raw_compat_is_enabled(DotDict(raw)):
        if set(active_channels) != _MOSID_UNIFIED_BLOCK_VIEW_CHANNELS:
            raise ValueError(f'raw compatibility requires active_channels {sorted(_MOSID_UNIFIED_BLOCK_VIEW_CHANNELS)!r}, got {active_channels!r}')
        ranking_fields = ('loss.raw_prefix_hard_negative_counts_by_lcp',)
        for dotted_key in ranking_fields:
            if not _has_nested(raw, dotted_key) or _get_nested(raw, dotted_key) is None:
                raise ValueError(f'missing required config field {dotted_key!r}')
        lcp_counts = _get_nested(raw, 'loss.raw_prefix_hard_negative_counts_by_lcp')
        if not lcp_counts:
            raise ValueError('loss.raw_prefix_hard_negative_counts_by_lcp must be a non-empty list')
        for depth, count in enumerate(lcp_counts):
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f'loss.raw_prefix_hard_negative_counts_by_lcp values must be integers, got depth {depth}: {count!r}')
            if count < 0:
                raise ValueError(f'loss.raw_prefix_hard_negative_counts_by_lcp values must be non-negative, got depth {depth}: {count}')

def _get_nested(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for key in dotted_key.split('.'):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current

def _has_nested(data: dict[str, Any], dotted_key: str) -> bool:
    current: Any = data
    for key in dotted_key.split('.'):
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True

def _validate_rrf_param_choices(param_choices: dict[str, Any]) -> None:
    allowed = {'k', 'rank_fusion_weight', 'raw_compat_weight'}
    for key, values in param_choices.items():
        if key not in allowed:
            raise ValueError(f'unknown rrf.param_choices.{key}')
        if not isinstance(values, list) or not values:
            raise ValueError(f'rrf.param_choices.{key} must be a non-empty list')
        for value in values:
            if isinstance(value, bool):
                raise TypeError(f'rrf.param_choices.{key} values must be numeric, got bool ({value!r})')
            number = float(value)
            if key == 'k':
                if number <= 0.0:
                    raise ValueError(f'rrf.param_choices.k values must be positive, got {value}')
            elif number < 0.0:
                raise ValueError(f'rrf.param_choices.{key} values must be non-negative, got {value}')
_REQUIRED_FIELDS = ['data.base_path', 'data.dataset', 'data.mode']

def _check_required(raw: dict[str, Any]) -> None:
    for dotted_key in _REQUIRED_FIELDS:
        if _get_nested(raw, dotted_key) is None:
            raise ValueError(f'missing required config field {dotted_key!r}')

def _build_paths(raw: dict[str, Any]) -> None:
    data = raw.get('data', {})
    base_path = data.get('base_path', '')
    dataset = data.get('dataset', '')
    mode = data.get('mode', '')
    data['_data_dir'] = os.path.join(base_path, dataset, mode)
_RUN_STAGES = {'rqvae', 'sasrec', 's3rec', 'fdsa', 'tiger', 'diger', 'diger_rqvae', 'unisrec', 'mosid', 'bert4rec', 'duorec', 'missrec', 'mmsr', 'dgmrec', 'i3mrec', 'mp4sr', 'mql4grec', 'macrec', 'fmlprec'}
_RUN_MODES = {'nm', 'sm'}
_RQVAE_MODALITIES = {'text', 'vision', 'collab', 'fused', 'unified'}
_TIGER_SID_SOURCES = {'text', 'vision', 'collab', 'fused', 'concat'}
_TIMESTAMP_PARTS = 5

def resolve_config_path(config_path: str | None, run_reference_path: str | os.PathLike[str] | None, fallback_config_path: str | None=None) -> str:
    if config_path is not None:
        return config_path
    run_config = find_run_config(run_reference_path)
    if run_config is not None:
        return str(run_config)
    if fallback_config_path is not None:
        return fallback_config_path
    raise ValueError('config path is required when checkpoint run config is unavailable; pass --config, keep config.yaml next to the checkpoint, or provide a fallback config')

def resolve_run_dir(run_reference_path: str | os.PathLike[str] | None) -> Path | None:
    if not run_reference_path:
        return None
    run_reference = Path(run_reference_path).expanduser()
    if run_reference.exists():
        return run_reference if run_reference.is_dir() else run_reference.parent
    return run_reference.parent if run_reference.suffix else run_reference

def find_run_config(run_reference_path: str | os.PathLike[str] | None) -> Path | None:
    run_dir = resolve_run_dir(run_reference_path)
    if run_dir is None:
        return None
    run_config = run_dir / 'config.yaml'
    if run_config.exists():
        return run_config
    return None

def describe_run_reference(run_reference_path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    run_dir = resolve_run_dir(run_reference_path)
    if run_dir is None:
        return None
    try:
        parts = _parse_run_dir_name(run_dir.name)
    except ValueError:
        return None
    info = dict(parts)
    info['run_reference_path'] = str(Path(run_reference_path).expanduser())
    info['run_dir'] = str(run_dir)
    run_config = find_run_config(run_dir)
    info['run_config'] = None if run_config is None else str(run_config)
    return info

def infer_run_overrides(run_reference_path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not run_reference_path:
        return {}
    run_parts = describe_run_reference(run_reference_path)
    if run_parts is None:
        return {}
    overrides: dict[str, Any] = {'data.dataset': run_parts['dataset'], 'data.mode': run_parts['mode']}
    stage = run_parts['stage']
    suffix = run_parts['suffix']
    if stage == 'rqvae':
        if suffix not in _RQVAE_MODALITIES:
            raise ValueError(f'cannot infer data.modality from run reference {run_reference_path!r}: unexpected rqvae suffix {suffix!r}')
        overrides['data.modality'] = suffix
    elif stage == 'tiger':
        if suffix not in _TIGER_SID_SOURCES:
            raise ValueError(f'cannot infer model.sid_source from run reference {run_reference_path!r}: unexpected tiger suffix {suffix!r}')
        overrides['model.sid_source'] = suffix
    return overrides

def infer_checkpoint_overrides(checkpoint_path: str | os.PathLike[str] | None) -> dict[str, Any]:
    return infer_run_overrides(checkpoint_path)

def _parse_run_dir_name(run_dir_name: str) -> dict[str, str]:
    tokens = run_dir_name.split('_')
    if len(tokens) < 3:
        raise ValueError(f'run dir name is too short: {run_dir_name!r}')
    timestamp = tokens[-_TIMESTAMP_PARTS:]
    core = tokens[:-_TIMESTAMP_PARTS] if all((part.isdigit() for part in timestamp)) else tokens
    mode_index = -1
    stage_index = -1
    stage_end_index = -1
    stage = ''
    for index in range(1, len(core) - 1):
        if core[index] not in _RUN_MODES:
            continue
        stage_tokens = core[index + 1:]
        for candidate in sorted(_RUN_STAGES, key=lambda value: len(value.split('_')), reverse=True):
            candidate_tokens = candidate.split('_')
            if stage_tokens[:len(candidate_tokens)] == candidate_tokens:
                mode_index = index
                stage_index = index + 1
                stage_end_index = stage_index + len(candidate_tokens)
                stage = candidate
                break
        if stage:
            break
    if mode_index < 1 or stage_index < 2 or stage_end_index < 3:
        raise ValueError(f'run dir name cannot determine dataset/mode/stage: {run_dir_name!r}')
    dataset = '_'.join(core[:mode_index])
    mode = core[mode_index]
    suffix_parts = core[stage_end_index:]
    suffix = '_'.join(suffix_parts)
    if stage == 'rqvae' and suffix_parts and (suffix_parts[0] in _RQVAE_MODALITIES):
        suffix = suffix_parts[0]
    elif stage == 'tiger' and suffix_parts and (suffix_parts[0] in _TIGER_SID_SOURCES):
        suffix = suffix_parts[0]
    if not dataset:
        raise ValueError(f'run dir name is missing dataset: {run_dir_name!r}')
    if mode not in _RUN_MODES:
        raise ValueError(f'run dir name contains unknown data mode {mode!r}: {run_dir_name!r}')
    return {'dataset': dataset, 'mode': mode, 'stage': stage, 'suffix': suffix}

def seed_run_dir(config: DotDict, stage: str) -> Path | None:
    """Identify unified artifacts, or opted-in separate runs, by training seeds."""
    if stage == 'rqvae' and getattr(config.data, 'modality', '') == 'unified':
        name = f'rqvae_unified_seed_{int(config.train.seed)}'
    elif stage == 'mosid' and getattr(config.model, 'sid_artifact_mode', '') == 'unified':
        stage1_seed = getattr(config.train, 'stage1_seed', None)
        metadata = Path(config.data._data_dir) / 'sid_unified_source.json'
        if metadata.is_file():
            source = json.loads(metadata.read_text(encoding='utf-8'))
            if stage1_seed is not None and int(stage1_seed) != int(source['seed']):
                raise ValueError('train.stage1_seed does not match exported SID source seed')
            stage1_seed = source['seed']
        if stage1_seed is None:
            raise ValueError('Stage 1 seed is unknown: re-export unified SID artifacts or pass --train.stage1_seed')
        name = f'mosid_seed_{int(stage1_seed)}-{int(config.train.seed)}'
    elif bool(getattr(config.train, 'seeded_run', False)) and stage == 'rqvae':
        name = f'rqvae_{config.data.modality}_seed_{int(config.train.seed)}'
    elif bool(getattr(config.train, 'seeded_run', False)) and stage == 'mosid':
        stage1_seed = getattr(config.train, 'stage1_seed', None)
        if stage1_seed is None:
            raise ValueError('Separate seeded runs require train.stage1_seed')
        name = f'mosid_separate_seed_{int(stage1_seed)}-{int(config.train.seed)}'
    else:
        return None
    return Path(config.train.output_dir) / f'{config.data.dataset}_{config.data.mode}_{name}'

def reusable_run(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    if (run_dir / 'best.pt').is_file() and (run_dir / 'config.yaml').is_file():
        return True
    if any(run_dir.iterdir()):
        raise FileExistsError(f'incomplete artifacts in {run_dir}; use --overwrite to retrain')
    return False

def build_run_dir(config: DotDict, stage: str, suffix: str='') -> str:
    seeded = seed_run_dir(config, stage)
    if seeded is not None:
        overwrite = bool(getattr(config, 'overwrite', False))
        if not overwrite:
            reusable_run(seeded)
        seeded.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for filename in ('best.pt', 'last.pt', 'config.yaml'):
                (seeded / filename).unlink(missing_ok=True)
        return str(seeded)
    timestamp = datetime.now().strftime('%m_%d_%H_%M_%S')
    parts = [config.data.dataset, config.data.mode, stage]
    if suffix:
        parts.append(suffix)
    parts.append(timestamp)
    run_dir = os.path.join(config.train.output_dir, '_'.join(parts))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def get_mod_presence_filename(config: DotDict) -> str:
    if config.data.mode == 'nm':
        return 'mod_presence.pt'
    probability = config.data.sm_prob if 'sm_prob' in config.data else 0.3
    return f'mod_presence_p{probability}.pt'
