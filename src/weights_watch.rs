//! Weights published by a trainer that runs alongside self-play.
//!
//! The trainer writes a pointer file holding `version path`; self-play checks it
//! at the start of each game and loads newer weights, so a game never mixes two
//! networks and no game waits for a load.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// The version and weights path in a pointer file, or None if it is missing or malformed.
pub fn read_published(pointer: &Path) -> Option<(u64, String)> {
    let text = std::fs::read_to_string(pointer).ok()?;
    let (version, path) = text.trim().split_once(' ')?;
    let path = path.trim();
    if path.is_empty() {
        return None;
    }
    Some((version.parse().ok()?, path.to_string()))
}

/// Write a pointer file through a temporary file and a rename, so readers see
/// either the old pointer or the new one.
pub fn write_published(pointer: &Path, version: u64, path: &str) -> std::io::Result<()> {
    let tmp = pointer.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp)?;
    writeln!(f, "{version} {path}")?;
    f.sync_all()?;
    std::fs::rename(&tmp, pointer)
}

type Loader<T> = Box<dyn Fn(&str) -> Result<T, String> + Send + Sync>;

/// A model that follows a pointer file, loading each newer version once.
pub struct WatchedModel<T> {
    pointer: PathBuf,
    loader: Loader<T>,
    current: Mutex<(u64, Arc<T>)>,
    /// The newest version tried, loaded or not, so a broken one is not retried.
    tried: Mutex<u64>,
}

impl<T> WatchedModel<T> {
    /// Load the weights the pointer names now; an error if there are none.
    pub fn new<F>(pointer: PathBuf, loader: F) -> Result<Self, String>
    where
        F: Fn(&str) -> Result<T, String> + Send + Sync + 'static,
    {
        let (version, path) = read_published(&pointer)
            .ok_or_else(|| format!("no published weights in {}", pointer.display()))?;
        let model = loader(&path)?;
        Ok(WatchedModel {
            pointer,
            loader: Box::new(loader),
            current: Mutex::new((version, Arc::new(model))),
            tried: Mutex::new(version),
        })
    }

    /// The newest loaded version and its model. If the pointer names a newer
    /// version, one caller loads it while the others keep the current model.
    pub fn current(&self) -> (u64, Arc<T>) {
        if let Some((version, path)) = read_published(&self.pointer) {
            if let Ok(mut tried) = self.tried.try_lock() {
                if version > *tried {
                    *tried = version;
                    match (self.loader)(&path) {
                        Ok(model) => *self.current.lock().unwrap() = (version, Arc::new(model)),
                        Err(e) => eprintln!("Keeping the current weights: {e}"),
                    }
                }
            }
        }
        let (version, model) = &*self.current.lock().unwrap();
        (*version, model.clone())
    }
}
