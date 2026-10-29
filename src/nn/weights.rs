//! Locating and loading PyTorch state dicts (safetensors or `.pth`).

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightsFile {
    Safetensors(PathBuf),
    Pth(PathBuf),
}

/// Find the state dict for a model path.
///
/// `.safetensors` and `.pth` paths are used as given. Any other path (usually a
/// TorchScript `gen_N.pt`, which candle cannot read) resolves to a sibling
/// `gen_N.safetensors`, then `gen_N.pth`, so existing scripts keep working.
pub fn resolve_weights(path: &Path) -> Result<WeightsFile, String> {
    // A symlink such as weights/latest.pt -> gen_N.pt: try siblings of the link
    // first, then siblings of its target.
    let target = std::fs::canonicalize(path).ok().filter(|t| t != path);
    std::iter::once(path)
        .chain(target.as_deref())
        .find_map(resolve_one)
        .ok_or_else(|| {
            format!(
                "No weights found for {}: expected a .safetensors or .pth state dict \
                 (TorchScript .pt files cannot be loaded)",
                path.display()
            )
        })
}

fn resolve_one(path: &Path) -> Option<WeightsFile> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "safetensors" => path.exists().then(|| WeightsFile::Safetensors(path.into())),
        "pth" => path.exists().then(|| WeightsFile::Pth(path.into())),
        _ => {
            let st = path.with_extension("safetensors");
            let pth = path.with_extension("pth");
            if st.exists() {
                Some(WeightsFile::Safetensors(st))
            } else if pth.exists() {
                Some(WeightsFile::Pth(pth))
            } else {
                None
            }
        }
    }
}

/// Load every tensor of the state dict for `path` onto `device`.
///
/// `.pth` files may be a training checkpoint (`{"model_state_dict": ...}`) or a
/// bare state dict.
pub fn load_state_dict(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>, String> {
    let err = |e: candle_core::Error| format!("Failed to read {}: {}", path.display(), e);
    match resolve_weights(path)? {
        WeightsFile::Safetensors(p) => candle_core::safetensors::load(&p, device).map_err(err),
        WeightsFile::Pth(p) => {
            let tensors = match candle_core::pickle::read_all_with_key(&p, Some("model_state_dict"))
            {
                Ok(t) if !t.is_empty() => t,
                _ => candle_core::pickle::read_all(&p).map_err(err)?,
            };
            tensors
                .into_iter()
                .map(|(name, t)| Ok((name, t.to_device(device).map_err(err)?)))
                .collect()
        }
    }
}
