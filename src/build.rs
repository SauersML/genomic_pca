// In your project root, create/edit build.rs

use std::env;

fn main() {
    // This tells Cargo to re-run this build script only if build.rs itself changes.
    println!("cargo:rerun-if-changed=build.rs");

    // Get the current build profile. Cargo sets this environment variable.
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    // We only want to apply these aggressive flags for RELEASE builds.
    if profile != "release" {
        eprintln!("[build.rs] Not a RELEASE build (profile: {}). SIMD target features will not be forced by this script. Defaults or std::simd scalar fallbacks will apply.", profile);
        return;
    }

    // Get the target architecture Cargo is building for.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|e| {
        eprintln!("[build.rs] CRITICAL WARNING: Could not determine target architecture (CARGO_CFG_TARGET_ARCH not set: {}). Cannot apply specific SIMD flags. Your SIMD code may run as slow scalar operations!", e);
        String::new() // Default to an empty string if not found, leading to no flags.
    });

    let mut rustc_flags = String::new();

    eprintln!("[build.rs] Configuring SIMD optimizations for RELEASE build targeting: {}", target_arch);

    match target_arch.as_str() {
        "x86_64" => {
            let features = [
                "avx",    // Base AVX
                "avx2",   // AVX2 for wider integer ops and more
                "fma",    // Fused Multiply-Add (very common with AVX/AVX2)
                "sse4.2", // Useful string/text processing and other instructions
                "popcnt", // Population count (count set bits)
            ];
            let feature_string = features.iter().map(|f| format!("+{}", f)).collect::<Vec<String>>().join(",");
            rustc_flags.push_str(&format!("-Ctarget-feature={}", feature_string));
            eprintln!("[build.rs]   x86_64: Enabling SIMD features: {}", feature_string);
        }
        "aarch64" => {
            let features = [
                "neon",      // Advanced SIMD (ASIMD)
                "fp-armv8",  // Standard ARMv8 floating point
            ];
            let feature_string = features.iter().map(|f| format!("+{}", f)).collect::<Vec<String>>().join(",");
            rustc_flags.push_str(&format!("-Ctarget-feature={}", feature_string));
            eprintln!("[build.rs]   aarch64: Enabling SIMD features: {}", feature_string);
        }
        _ => {
            eprintln!("[build.rs]   WARN: Unknown or unsupported target_arch '{}' for specific SIMD flags in this script. Relying on compiler defaults. `std::simd` may use scalar fallbacks.", target_arch);
        }
    }

    if !rustc_flags.is_empty() {
        // Apply the determined flags.
        println!("cargo:rustc-flags={}", rustc_flags);
    }
    eprintln!("[build.rs] Script finished. If SIMD flags were set, subsequent compilation of your crate will use them.");
}
