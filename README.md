[IMPORTANT] 
That Fork is mostly Vibe Coded. Runs on my system without any problems. Stripped away a lot of stuff that isnt compatible or required for ROCm and fixed a few things. Using a 7900XT lora Training with 2.5s/it
Prerequisites

    Python 3.10+ (Python 3.12 recommended)
    Node.js (Required for the Web UI)
    Adrenalin Driver Version 26.6.1 or newer

Installation
1. Clone the repository or download the zip (Green "<> Code" button)
2. Check if right ROCm Version is used (GFX110x(RX 7XXX XT/X GPU) is default no changes needed)

For everything else check https://github.com/CS1o/Stable-Diffusion-Info/wiki/Lora-Trainer-Setup-Guides#amd-onetrainer-with-rocmtherock and edit the setup_env.bat
3. Open folder training-ui and start OR edit start_training_ui_anima.bat

Because i have 2 grafic chips the default HIP Device is set to 1 (set HIP_VISIBLE_DEVICES=1) If you have 1 device change to 0 if you have more Change to the device you want to use. For everything else check Original Repo.