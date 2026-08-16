use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use crossbeam_channel::{unbounded, Sender};
use crate::core::state::CompactTick;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RecordedTick {
    pub coin_id: usize,
    pub tick: CompactTick,
}

pub struct TickRecorder {
    sender: Sender<RecordedTick>,
}

impl TickRecorder {
    pub fn new(file_path: &str) -> Self {
        let (tx, rx) = unbounded::<RecordedTick>();
        let path = file_path.to_string();
        
        // Create directory if it doesn't exist
        if let Some(parent) = Path::new(&path).parent() {
            std::fs::create_dir_all(parent).unwrap_or_default();
        }
        
        std::thread::spawn(move || {
            let mut opts = OpenOptions::new();
            opts.create(true).append(true);
            
            #[cfg(target_os = "windows")]
            {
                use std::os::windows::fs::OpenOptionsExt;
                opts.share_mode(7); // FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
            }
            
            let file = opts.open(&path)
                .expect("Failed to open tick recorder file");
            
            // 16MB buffer to minimize disk I/O blocks
            let mut writer = BufWriter::with_capacity(1024 * 1024 * 16, file); 
            
            let mut flush_counter = 0;
            
            while let Ok(record) = rx.recv() {
                // Transmute memory to bytes to write zero-copy
                let bytes: [u8; 48] = unsafe { std::mem::transmute(record) };
                if let Err(e) = writer.write_all(&bytes) {
                    eprintln!("[TickRecorder] Write Error: {}", e);
                }
                
                flush_counter += 1;
                if flush_counter >= 100_000 {
                    let _ = writer.flush();
                    flush_counter = 0;
                }
            }
            let _ = writer.flush();
        });
        
        Self { sender: tx }
    }
    
    #[inline(always)]
    pub fn record(&self, coin_id: usize, tick: CompactTick) {
        let record = RecordedTick {
            coin_id,
            tick,
        };
        let _ = self.sender.send(record);
    }
}
