use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:warning=Running set_version.py");
    let status = Command::new("python3")
        .args(["../set_version.py"])
        .status()
        .expect("Failed to start set_version.py");
    if !status.success() {
        panic!("set_version.py failed with status: {}", status);
    }
}
