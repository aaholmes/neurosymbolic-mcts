//! Thread placement and per-thread CPU accounting: CPU lists, pinning a thread
//! to chosen logical CPUs, and the CPU time the calling thread has used.

use crate::mcts::tactical_mcts::TimingAccumulator;
use std::time::{Duration, Instant};

/// Parse a CPU list such as "0,16" or "2-4,8" into sorted, distinct CPU numbers.
pub fn parse_cpu_list(s: &str) -> Result<Vec<usize>, String> {
    let mut cpus = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        let num = |t: &str| {
            t.trim()
                .parse::<usize>()
                .map_err(|_| format!("bad CPU number {t:?} in {s:?}"))
        };
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (num(a)?, num(b)?);
                if a > b {
                    return Err(format!("descending CPU range {part:?} in {s:?}"));
                }
                cpus.extend(a..=b);
            }
            None => cpus.push(num(part)?),
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    Ok(cpus)
}

/// Restrict the calling thread to the given logical CPUs.
#[cfg(target_os = "linux")]
pub fn pin_current_thread(cpus: &[usize]) -> std::io::Result<()> {
    if cpus.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "cannot pin a thread to no CPUs",
        ));
    }
    // SAFETY: cpu_set_t is plain data; CPU_ZERO/CPU_SET only write within it.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cpus {
            libc::CPU_SET(c, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

/// The logical CPUs the calling thread may run on.
#[cfg(target_os = "linux")]
pub fn current_thread_cpus() -> std::io::Result<Vec<usize>> {
    // SAFETY: as in pin_current_thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok((0..libc::CPU_SETSIZE as usize)
            .filter(|&c| libc::CPU_ISSET(c, &set))
            .collect())
    }
}

/// CPU time used so far by the calling thread (zero where unsupported).
pub fn thread_cpu_time() -> Duration {
    #[cfg(unix)]
    {
        // SAFETY: clock_gettime writes one timespec.
        let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
        if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) } == 0 {
            return Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32);
        }
    }
    Duration::ZERO
}

/// Wall time and the thread's own CPU time spent in one phase of work. The
/// difference is time the thread was off its core: waiting to run, or blocked.
#[derive(Debug, Clone, Default)]
pub struct PhaseTiming {
    pub wall: TimingAccumulator,
    pub cpu: Duration,
}

/// A named range on Nsight Systems timelines (with the `nvtx` feature), ended
/// when dropped. Without the feature it does nothing.
pub struct Span {
    #[cfg(feature = "nvtx")]
    _range: nvtx::LocalRange,
}

impl Drop for Span {
    /// Ends the span (the NVTX range, when enabled, ends as its field drops).
    fn drop(&mut self) {}
}

/// Open a named span on the calling thread.
pub fn span(name: &'static std::ffi::CStr) -> Span {
    #[cfg(not(feature = "nvtx"))]
    let _ = name;
    Span {
        #[cfg(feature = "nvtx")]
        _range: nvtx::LocalRange::new(name),
    }
}

/// Measures one occurrence of a phase, and marks it as a span:
/// `let c = PhaseClock::start(c"name"); ...; c.stop(&mut p)`.
pub struct PhaseClock {
    wall: Instant,
    cpu: Duration,
    _span: Span,
}

impl PhaseClock {
    pub fn start(name: &'static std::ffi::CStr) -> Self {
        Self {
            _span: span(name),
            wall: Instant::now(),
            cpu: thread_cpu_time(),
        }
    }

    /// Record the time since `start` in `phase`, and return the wall time.
    pub fn stop(self, phase: &mut PhaseTiming) -> Duration {
        let wall = self.wall.elapsed();
        phase.wall.record(wall);
        phase.cpu += thread_cpu_time().saturating_sub(self.cpu);
        wall
    }
}

/// The CPUs in `all` that are not in `reserved`; all of them if none would remain.
pub fn other_cpus(all: &[usize], reserved: &[usize]) -> Vec<usize> {
    let rest: Vec<usize> = all
        .iter()
        .copied()
        .filter(|c| !reserved.contains(c))
        .collect();
    if rest.is_empty() {
        all.to_vec()
    } else {
        rest
    }
}

/// Build the global rayon pool for game threads, named `kf-game-{i}`. With
/// `reserved` CPUs (the inference threads'), each game thread is pinned to
/// the remaining ones. `threads` falls back to RAYON_NUM_THREADS, then rayon's
/// default.
pub fn build_game_pool(threads: Option<usize>, reserved: Option<&[usize]>) {
    let threads = threads.or_else(|| {
        std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
    });
    let mut builder = rayon::ThreadPoolBuilder::new().thread_name(|i| format!("kf-game-{i}"));
    if let Some(n) = threads {
        builder = builder.num_threads(n);
    }
    #[cfg(target_os = "linux")]
    if let Some(reserved) = reserved {
        let all = current_thread_cpus().unwrap_or_default();
        let game_cpus = other_cpus(&all, reserved);
        builder = builder.start_handler(move |_| {
            if let Err(e) = pin_current_thread(&game_cpus) {
                eprintln!("[game] could not pin to CPUs: {e}");
            }
        });
    }
    #[cfg(not(target_os = "linux"))]
    let _ = reserved;
    builder.build_global().ok(); // fails silently if already built
}
