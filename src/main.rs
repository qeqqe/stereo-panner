use std::io::{stdout, Write};
use std::net::UdpSocket;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};


// smoothing: higher = smoother but more latency (0.0 - 0.99)
const SMOOTHING_FACTOR: f64 = 0.65;

// min time between updates (20ms = ~50fps)
const UPDATE_RATE_MS: u64 = 20;

// only send command if angle changes by this many degrees
const CHANGE_THRESHOLD: f64 = 0.5;

// default radius, can change at runtime
const DEFAULT_RADIUS: f64 = 1.5;
const MIN_RADIUS: f64 = 0.1;
const MAX_RADIUS: f64 = 10.0;
const RADIUS_STEP: f64 = 0.1;

// dynamic reverb wet/dry mix depending on distance
const MIN_REVERB: f64 = 0.05;  // closest
const MAX_REVERB: f64 = 0.60;  // farthest

// speaker angles for front and back modes (base angles at 100% width)
const FRONT_LEFT_ANGLE: f64 = 45.0;   // +45° (front-left) - wider for less focus
const FRONT_RIGHT_ANGLE: f64 = -45.0; // -45° (front-right)
const BACK_LEFT_ANGLE: f64 = 135.0;   // +135° (back-left)
const BACK_RIGHT_ANGLE: f64 = -135.0; // -135° (back-right)

// stereo width control: adjusts speaker separation
const DEFAULT_WIDTH: f64 = 1.0;  // 100% = full separation
const MIN_WIDTH: f64 = 0.3;      // 30% = narrow (more focused)
const MAX_WIDTH: f64 = 1.5;      // 150% = extra wide (very diffuse)
const WIDTH_STEP: f64 = 0.1;

// node name to search for in pipewire
const SPATIALIZER_NODE_NAME: &str = "effect_input.spatializer";

// ==============================================================================
// DATA STRUCTURES
// ==============================================================================

#[derive(Clone, Copy, PartialEq)]
enum SpeakerMode {
    Front,
    Back,
}

impl SpeakerMode {
    fn label(&self) -> &'static str {
        match self {
            SpeakerMode::Front => "FRONT",
            SpeakerMode::Back => "BACK",
        }
    }

    fn base_angles(&self) -> (f64, f64) {
        match self {
            SpeakerMode::Front => (BACK_LEFT_ANGLE, BACK_RIGHT_ANGLE),
            SpeakerMode::Back => (FRONT_LEFT_ANGLE, FRONT_RIGHT_ANGLE),
        }
    }
}

struct SmoothedState {
    yaw: f64,
    pitch: f64,
    roll: f64,
}

impl SmoothedState {
    fn new() -> Self {
        Self { yaw: 0.0, pitch: 0.0, roll: 0.0 }
    }

    // apply exponential smoothing
    fn update(&mut self, raw_yaw: f64, raw_pitch: f64, raw_roll: f64) {
        self.yaw = SMOOTHING_FACTOR * self.yaw + (1.0 - SMOOTHING_FACTOR) * raw_yaw;
        self.pitch = SMOOTHING_FACTOR * self.pitch + (1.0 - SMOOTHING_FACTOR) * raw_pitch;
        self.roll = SMOOTHING_FACTOR * self.roll + (1.0 - SMOOTHING_FACTOR) * raw_roll;
    }
}

// holds the calculated positions for the virtual speakers relative to head
struct SpatialState {
    left_az: f64,
    right_az: f64,
    elevation: f64,
    radius: f64,
    gain: f64, // volume scaling based on radius (1.0 / radius)
    reverb_gain: f64, // wet signal amount (0.0 - 1.0)
}

impl SpatialState {
    fn from_head_tracking(yaw: f64, pitch: f64, radius: f64, mode: SpeakerMode, reverb_enabled: bool, width: f64) -> Self {
        // get base speaker angles based on mode
        let (left_base, right_base) = mode.base_angles();

        // width > 1.0 = wider (diffused), width < 1.0 = narrower (focused)
        let left_base_scaled = left_base * width;
        let right_base_scaled = right_base * width;

        // relative azimuth = base_pos - head_yaw
        let left_az = left_base_scaled - yaw;
        let right_az = right_base_scaled - yaw;

        // pitch is inverted (looking up moves the source down relative to eyes)
        let elevation = -pitch;

        // calculate gain: inverse relationship with radius
        // at radius 1.0 = 100% gain, radius 2.0 = 50% gain, etc.
        // clamp to reasonable range
        let gain = (1.0 / radius).clamp(0.1, 2.0);

        // calculate reverb gain using square-root curve for natural progression
        // sqrt gives more reverb early on, then tapers - matches physical acoustics
        let reverb_gain = if reverb_enabled {
            let normalized = ((radius - MIN_RADIUS) / (MAX_RADIUS - MIN_RADIUS)).clamp(0.0, 1.0);
            MIN_REVERB + normalized.sqrt() * (MAX_REVERB - MIN_REVERB)
        } else {
            0.0 // reverb disabled
        };

        Self { left_az, right_az, elevation, radius, gain, reverb_gain }
    }
}

// ==============================================================================
// DISPLAY HELPERS
// ==============================================================================

fn clear_screen() {
    stdout()
        .execute(Clear(ClearType::All))
        .ok();
    stdout()
        .execute(cursor::MoveTo(0, 0))
        .ok();
}

// helper: calculate string width ignoring ansi color codes
// fixes border alignment by counting emojis as 2 width
fn get_visible_width(s: &str) -> usize {
    let mut width = 0;
    let mut inside_ansi = false;
    for c in s.chars() {
        if c == '\x1B' {
            inside_ansi = true;
            continue;
        }
        if inside_ansi {
            if c == 'm' {
                inside_ansi = false;
            }
            continue;
        }
        // account for double-width emojis used in headers
        match c {
             '🎧' | '🧭' | '🔊' | '📐' |  '📡' | '📈'  => width += 2,
            _ => width += 1,
        }
    }
    width
}

// render an azimuth position bar showing where a speaker is relative to center
fn render_azimuth_bar(azimuth: f64, width: usize) -> String {
    let mut bar = String::with_capacity(width + 20);
    bar.push('[');

    // map azimuth (-180..180) to bar position
    // clamp to reasonable range for display
    let clamped = azimuth.clamp(-90.0, 90.0);
    let normalized = (clamped + 90.0) / 180.0; // 0..1
    let pos = (normalized * (width - 1) as f64).round() as usize;
    let center_idx = width / 2;

    for i in 0..width {
        if i == pos {
            bar.push_str("\x1B[1;33m◆\x1B[0m"); // speaker position marker
        } else if i == center_idx {
            bar.push_str("\x1B[90m│\x1B[0m"); // center line
        } else {
            bar.push(' ');
        }
    }

    bar.push(']');
    bar
}

// render an elevation indicator
fn render_elevation_indicator(elevation: f64) -> &'static str {
    if elevation > 10.0 {
        "⬆ Above"
    } else if elevation < -10.0 {
        "⬇ Below"
    } else {
        "━ Level"
    }
}

fn render_dashboard(
    smoothed: &SmoothedState,
    raw_yaw: f64,
    raw_pitch: f64,
    raw_roll: f64,
    spatial: &SpatialState,
    fps: f64,
    node_id: &Option<String>,
    latency_ms: f64,
    packets: u64,
    mode: SpeakerMode,
    reverb_enabled: bool,
    width: f64,
) {
    clear_screen();

    let draw_row = |content: &str| {
        let inner_target = 66;
        let visible = get_visible_width(content);
        let padding = if inner_target > visible { inner_target - visible } else { 0 };
        print!("\x1B[1;96m║\x1B[0m{}{}\x1B[1;96m║\x1B[0m\r\n", content, " ".repeat(padding));
    };

    let pad_field = |text: String, width: usize| -> String {
        let vis = get_visible_width(&text);
        let p = if width > vis { width - vis } else { 0 };
        format!("{}{}", text, " ".repeat(p))
    };

    print!("\x1B[1;96m╔══════════════════════════════════════════════════════════════════╗\x1B[0m\r\n");

    let title = "\x1B[1;37m🎧 SPATIAL AUDIO ENGINE (HRTF STEREO)\x1B[0m";
    let t_vis = get_visible_width(title);
    let t_pad = (66 - t_vis) / 2;
    print!("\x1B[1;96m║\x1B[0m{}{}{}\x1B[1;96m║\x1B[0m\r\n", " ".repeat(t_pad), title, " ".repeat(66 - t_vis - t_pad));
    print!("\x1B[1;96m╠══════════════════════════════════════════════════════════════════╣\x1B[0m\r\n");

    draw_row(&format!("  {}", "\x1B[1;33m🧭 HEAD TRACKING\x1B[0m"));
    draw_row("");
    draw_row(&format!("    \x1B[90mRAW:\x1B[0m     Yaw={:>7.1}°  Pitch={:>7.1}°  Roll={:>7.1}°",
                      raw_yaw, raw_pitch, raw_roll));
    draw_row(&format!("    \x1B[1;37mSMOOTH:\x1B[0m  Yaw={:>7.1}°  Pitch={:>7.1}°  Roll={:>7.1}°",
                      smoothed.yaw, smoothed.pitch, smoothed.roll));

    draw_row("");
    print!("\x1B[1;96m╠══════════════════════════════════════════════════════════════════╣\x1B[0m\r\n");

    let mode_color = match mode {
        SpeakerMode::Front => "\x1B[1;32m",
        SpeakerMode::Back => "\x1B[1;33m",
    };
    draw_row(&format!("  \x1B[1;35m🔊 VIRTUAL SPEAKERS\x1B[0m  [{}{}°\x1B[0m]", mode_color, mode.label()));
    draw_row("");

    let adjust_display_azimuth = |a: f64| -> f64 {
        let mut x = a;
        // normalize to -180..180
        while x <= -180.0 { x += 360.0; }
        while x > 180.0 { x -= 360.0; }
        // so it doesnt clamp to the end
        if x > 90.0 {
            x -= 180.0;
        } else if x < -90.0 {
            x += 180.0;
        }
        x
    };

    let left_display = adjust_display_azimuth(spatial.right_az);
    let right_display = adjust_display_azimuth(spatial.left_az);

    let l_bar = render_azimuth_bar(left_display, 24);
    draw_row(&format!("    \x1B[1;34mLeft Speaker:\x1B[0m  {}  {:>+6.1}°", l_bar, left_display));

    let r_bar = render_azimuth_bar(right_display, 24);
    draw_row(&format!("    \x1B[1;35mRight Speaker:\x1B[0m {}  {:>+6.1}°", r_bar, right_display));

    draw_row("");

    let elev_indicator = render_elevation_indicator(spatial.elevation);
    draw_row(&format!("    \x1B[1;37mElevation:\x1B[0m {:>+6.1}°  {}", spatial.elevation, elev_indicator));

    let gain_pct = spatial.gain * 100.0;
    draw_row(&format!("    \x1B[1;37mRadius:\x1B[0m    {:>6.2}m  (Gain: {:>3.0}%)", spatial.radius, gain_pct));

    let reverb_pct = spatial.reverb_gain * 100.0;
    let reverb_status = if reverb_enabled { "\x1B[1;32mON\x1B[0m" } else { "\x1B[1;31mOFF\x1B[0m" };
    draw_row(&format!("    \x1B[1;37mReverb:\x1B[0m   {:>6.1}%  [{}]", reverb_pct, reverb_status));

    draw_row("");
    print!("\x1B[1;96m╠══════════════════════════════════════════════════════════════════╣\x1B[0m\r\n");

    draw_row(&format!("  {}", "\x1B[1;33m📐 STEREO FIELD\x1B[0m"));
    draw_row("");

    let width_pct = width * 100.0;
    let width_desc = if width >= 1.2 {
        "\x1B[1;36mVery Wide\x1B[0m"
    } else if width >= 0.8 {
        "\x1B[1;37mNormal\x1B[0m"
    } else {
        "\x1B[1;33mNarrow\x1B[0m"
    };
    draw_row(&format!("    \x1B[1;37mWidth:\x1B[0m    {:>6.0}%  ({})", width_pct, width_desc));

    let sep_angle = (spatial.left_az - spatial.right_az).abs();
    draw_row(&format!("    \x1B[1;37mSeparation:\x1B[0m {:>5.1}°  (speaker spread)", sep_angle));

    draw_row("");
    print!("\x1B[1;96m╠══════════════════════════════════════════════════════════════════╣\x1B[0m\r\n");

    draw_row(&format!("  {}", "\x1B[1;32m📡 CONNECTION\x1B[0m"));
    draw_row("");

    let status = match node_id {
        Some(id) => format!("\x1B[1;32m✓ LINKED\x1B[0m to Node \x1B[1;37m{}\x1B[0m ({})", id, SPATIALIZER_NODE_NAME),
        None => format!("\x1B[1;31m✗ SEARCHING\x1B[0m for '{}'...", SPATIALIZER_NODE_NAME),
    };
    draw_row(&format!("    {}", status));

    draw_row("");
    print!("\x1B[1;96m╠══════════════════════════════════════════════════════════════════╣\x1B[0m\r\n");

    draw_row(&format!("  {}", "\x1B[1;34m📈 STATS\x1B[0m"));
    draw_row("");

    let col_width = 25;

    let fps_str = pad_field(format!("FPS: \x1B[1;37m{:>5.1}\x1B[0m", fps), col_width);
    let lat_str = format!("Latency: \x1B[1;37m{:>5.2}ms\x1B[0m", latency_ms);
    draw_row(&format!("    {}  │  {}", fps_str, lat_str));

    let pkts_str = pad_field(format!("Packets: \x1B[1;37m{}\x1B[0m", packets), col_width);
    let thresh_str = format!("Threshold: \x1B[1;37m{:.1}°\x1B[0m", CHANGE_THRESHOLD);
    draw_row(&format!("    {}  │  {}", pkts_str, thresh_str));

    let smooth_str = pad_field(format!("Smoothing: \x1B[1;37m{:.0}%\x1B[0m", SMOOTHING_FACTOR * 100.0), col_width);
    draw_row(&format!("    {}  │", smooth_str));

    draw_row("");
    print!("\x1B[1;96m╠══════════════════════════════════════════════════════════════════╣\x1B[0m\r\n");

    draw_row(&format!("  {}", "\x1B[1;90m⌨ CONTROLS\x1B[0m"));
    draw_row("    \x1B[90m↑/↓\x1B[0m Radius   \x1B[90m←/→\x1B[0m Width   \x1B[90mW\x1B[0m Front   \x1B[90mS\x1B[0m Back");
    draw_row("    \x1B[90mR\x1B[0m Reverb   \x1B[90mQ/Esc\x1B[0m Quit");
    print!("\x1B[1;96m╚══════════════════════════════════════════════════════════════════╝\x1B[0m\r\n");
}

// ==============================================================================
// PIPEWIRE CONTROL
// ==============================================================================

fn find_spatializer_node() -> Option<String> {
        // run 'pw-cli ls Node'
    let output = Command::new("pw-cli").args(["ls", "Node"]).output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);

    let mut current_id = String::new();

    // simple state machine parser (no external deps)
    for line in text.lines() {
        let trim = line.trim();
        if trim.starts_with("id") {
            // "id 36, type PipeWire:Interface:Node..."
            if let Some(id_part) = trim.split_whitespace().nth(1) {
                current_id = id_part.trim_matches(',').to_string();
            }
        }
        // check for our target node name
        if trim.contains("node.name") && trim.contains(SPATIALIZER_NODE_NAME) {
            return Some(current_id);
        }
    }
    None
}

fn update_pipewire(id: &str, spatial: &SpatialState) {
    // build the json for the stereo filter-chain
    // sets params for both 'spat_left' and 'spat_right' nodes
    // uses dynamic radius and includes gain for reverb simulation
    let dry_gain = 1.0 - spatial.reverb_gain;
    let json_payload = format!(
        "{{ \"params\": [ \
            \"spat_left:Azimuth\", {:.2}, \
            \"spat_left:Elevation\", {:.2}, \
            \"spat_left:Radius\", {:.2}, \
            \"spat_left:Gain\", {:.2}, \
            \"spat_right:Azimuth\", {:.2}, \
            \"spat_right:Elevation\", {:.2}, \
            \"spat_right:Radius\", {:.2}, \
            \"spat_right:Gain\", {:.2}, \
            \"final_mix_l:Gain 1\", {:.2}, \
            \"final_mix_l:Gain 2\", {:.2}, \
            \"final_mix_r:Gain 1\", {:.2}, \
            \"final_mix_r:Gain 2\", {:.2} \
        ] }}",
        spatial.left_az, spatial.elevation, spatial.radius, spatial.gain,
        spatial.right_az, spatial.elevation, spatial.radius, spatial.gain,
        dry_gain, spatial.reverb_gain,
        dry_gain, spatial.reverb_gain
    );

    // spawn async (fire and forget) to prevent frame drops
    // redirect stdout/stderr to null to prevent tui artifacts
    Command::new("pw-cli")
        .args(["set-param", id, "Props", &json_payload])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok();
}

// ==============================================================================
// MAIN
// ==============================================================================

fn main() {
    // enable raw mode for keyboard input
    terminal::enable_raw_mode().expect("Failed to enable raw mode");
    stdout().execute(EnterAlternateScreen).expect("Failed to enter alternate screen");

    // make sure we cleanup on exit
    let result = run_main_loop();

    // cleanup terminal
    terminal::disable_raw_mode().ok();
    stdout().execute(LeaveAlternateScreen).ok();

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_main_loop() -> Result<(), String> {
    clear_screen();
    print!("\x1B[1;96m╔══════════════════════════════════════════════════════════════════╗\x1B[0m\r\n");
    print!("\x1B[1;96m║\x1B[0m{:^66}\x1B[1;96m║\x1B[0m\r\n", "\x1B[1;37m🎧 SPATIAL AUDIO ENGINE\x1B[0m");
    print!("\x1B[1;96m╠══════════════════════════════════════════════════════════════════╣\x1B[0m\r\n");
    print!("\x1B[1;96m║\x1B[0m{:66}\x1B[1;96m║\x1B[0m\r\n", "");
    print!("\x1B[1;96m║\x1B[0m  {:<64}\x1B[1;96m║\x1B[0m\r\n", "🔌 Binding to UDP port 4242...");
    stdout().flush().ok();

    let socket = match UdpSocket::bind("127.0.0.1:4242") {
        Ok(s) => {
            print!("\x1B[1;96m║\x1B[0m  {:<64}\x1B[1;96m║\x1B[0m\r\n", "\x1B[1;32m✓ Socket bound successfully!\x1B[0m");
            s
        }
        Err(e) => {
            return Err(format!("Failed to bind socket: {}", e));
        }
    };

    socket.set_read_timeout(Some(Duration::from_millis(10))).ok();

    print!("\x1B[1;96m║\x1B[0m{:66}\x1B[1;96m║\x1B[0m\r\n", "");
    print!("\x1B[1;96m║\x1B[0m  {:<64}\x1B[1;96m║\x1B[0m\r\n",
             format!("🔍 Searching for '{}'...", SPATIALIZER_NODE_NAME));
    print!("\x1B[1;96m║\x1B[0m  {:<64}\x1B[1;96m║\x1B[0m\r\n", "\x1B[1;33m⏳ Waiting for OpenTrack data...\x1B[0m");
    print!("\x1B[1;96m║\x1B[0m     {:<61}\x1B[1;96m║\x1B[0m\r\n", "Make sure OpenTrack is sending UDP to 127.0.0.1:4242");
    print!("\x1B[1;96m║\x1B[0m{:66}\x1B[1;96m║\x1B[0m\r\n", "");
    print!("\x1B[1;96m╚══════════════════════════════════════════════════════════════════╝\x1B[0m\r\n");
    stdout().flush().ok();

    let mut buf = [0u8; 48];
    let mut smoothed = SmoothedState::new();

    // state tracking
    let mut cached_node_id: Option<String> = None;
    let mut last_node_search = Instant::now();
    let mut last_update_time = Instant::now();

    // fps calculation
    let mut frame_count: u32 = 0;
    let mut last_fps_calc = Instant::now();
    let mut current_fps: f64 = 0.0;

    // packet counter
    let mut packet_count: u64 = 0;

    // don't spam pipewire if head hasn't moved
    let mut last_sent_yaw: f64 = f64::MAX;
    let mut last_sent_pitch: f64 = f64::MAX;
    let mut last_sent_radius: f64 = f64::MAX;

    // latency tracking
    let mut latency_samples: Vec<f64> = Vec::with_capacity(30);
    let mut avg_latency_ms: f64 = 0.0;

    // raw values for display (set on first packet)
    let (mut raw_yaw, mut raw_pitch, mut raw_roll): (f64, f64, f64);

    // dynamic state: radius, speaker mode, and width
    let mut current_radius: f64 = DEFAULT_RADIUS;
    let mut speaker_mode: SpeakerMode = SpeakerMode::Front;
    let mut reverb_enabled: bool = false; // off by default
    let mut current_width: f64 = DEFAULT_WIDTH;

    // flag to force update when user changes settings
    let mut force_update = false;

    loop {
        // 1. handle keyboard input (non-blocking)
        if event::poll(Duration::from_secs(0)).unwrap_or(false) {
            if let Ok(Event::Key(key_event)) = event::read() {
                match handle_key_event(key_event, &mut current_radius, &mut speaker_mode, &mut reverb_enabled, &mut current_width) {
                    KeyAction::Quit => break,
                    KeyAction::Changed => {
                        force_update = true;
                    }
                    KeyAction::None => {}
                }
            }
        }

        // 2. periodically search for node id if not found
        if cached_node_id.is_none() && last_node_search.elapsed().as_secs() > 2 {
            cached_node_id = find_spatializer_node();
            last_node_search = Instant::now();
        }

        // 3. read udp packet
        match socket.recv_from(&mut buf) {
            Ok((48, _)) => {
                packet_count += 1;

                // parse opentrack data: [x, y, z, yaw, pitch, roll] as f64
                let data: [f64; 6] = unsafe { std::mem::transmute(buf) };
                raw_yaw = data[3];
                raw_pitch = data[4];
                raw_roll = data[5];

                // apply smoothing
                smoothed.update(raw_yaw, raw_pitch, raw_roll);

                // 4. rate limit updates
                if last_update_time.elapsed() < Duration::from_millis(UPDATE_RATE_MS) && !force_update {
                    continue;
                }

                // calculate spatial positions with current radius, mode, and width
                let spatial = SpatialState::from_head_tracking(
                    smoothed.yaw,
                    smoothed.pitch,
                    current_radius,
                    speaker_mode,
                    reverb_enabled,
                    current_width,
                );

                // 5. send to pipewire (only if changed enough to avoid spamming, or forced)
                if let Some(ref id) = cached_node_id {
                    let yaw_changed = (smoothed.yaw - last_sent_yaw).abs() > CHANGE_THRESHOLD;
                    let pitch_changed = (smoothed.pitch - last_sent_pitch).abs() > CHANGE_THRESHOLD;
                    let radius_changed = (current_radius - last_sent_radius).abs() > 0.01;

                    if yaw_changed || pitch_changed || radius_changed || force_update {
                        let start = Instant::now();
                        update_pipewire(id, &spatial);
                        let cmd_latency = start.elapsed().as_secs_f64() * 1000.0;

                        // track latency samples for averaging
                        latency_samples.push(cmd_latency);
                        if latency_samples.len() > 30 {
                            latency_samples.remove(0);
                        }
                        avg_latency_ms = latency_samples.iter().sum::<f64>() / latency_samples.len() as f64;

                        last_sent_yaw = smoothed.yaw;
                        last_sent_pitch = smoothed.pitch;
                        last_sent_radius = current_radius;
                    }
                }

                force_update = false;

                // 6. fps calculation
                frame_count += 1;
                if last_fps_calc.elapsed() >= Duration::from_secs(1) {
                    current_fps = frame_count as f64 / last_fps_calc.elapsed().as_secs_f64();
                    frame_count = 0;
                    last_fps_calc = Instant::now();
                }

                // 7. render dashboard
                render_dashboard(
                    &smoothed,
                    raw_yaw,
                    raw_pitch,
                    raw_roll,
                    &spatial,
                    current_fps,
                    &cached_node_id,
                    avg_latency_ms,
                    packet_count,
                    speaker_mode,
                    reverb_enabled,
                    current_width,
                );
                stdout().flush().ok();

                last_update_time = Instant::now();
            }
            Ok(_) => continue, // bad packet size, skip
            Err(e) => {
                if e.kind() != std::io::ErrorKind::WouldBlock
                    && e.kind() != std::io::ErrorKind::TimedOut {
                    // don't print errors in raw mode, just continue
                }
                // sleep a tiny bit to save cpu when no data
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    Ok(())
}

// ==============================================================================
// keyboard handling
// ==============================================================================

enum KeyAction {
    Quit,
    Changed,
    None,
}

fn handle_key_event(
    key: KeyEvent,
    radius: &mut f64,
    mode: &mut SpeakerMode,
    reverb_enabled: &mut bool,
    width: &mut f64,
) -> KeyAction {
    match key.code {
        // quit keys
        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => KeyAction::Quit,
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => KeyAction::Quit,

        // radius control: up/down arrows
        KeyCode::Up => {
            *radius = (*radius + RADIUS_STEP).min(MAX_RADIUS);
            KeyAction::Changed
        }
        KeyCode::Down => {
            *radius = (*radius - RADIUS_STEP).max(MIN_RADIUS);
            KeyAction::Changed
        }

        // width control: left/right arrows
        KeyCode::Right => {
            *width = (*width + WIDTH_STEP).min(MAX_WIDTH);
            KeyAction::Changed
        }
        KeyCode::Left => {
            *width = (*width - WIDTH_STEP).max(MIN_WIDTH);
            KeyAction::Changed
        }

        // speaker mode: w = front, s = back
        KeyCode::Char('w') | KeyCode::Char('W') => {
            if *mode != SpeakerMode::Front {
                *mode = SpeakerMode::Front;
                KeyAction::Changed
            } else {
                KeyAction::None
            }
        }
        KeyCode::Char('s') | KeyCode::Char('S') => {
            if *mode != SpeakerMode::Back {
                *mode = SpeakerMode::Back;
                KeyAction::Changed
            } else {
                KeyAction::None
            }
        }

        // reverb toggle: r key
        KeyCode::Char('r') | KeyCode::Char('R') => {
            *reverb_enabled = !*reverb_enabled;
            KeyAction::Changed
        }

        _ => KeyAction::None,
    }
}