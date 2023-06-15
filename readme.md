[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Muse tools
This repository contains scripts to work with the Muse2 EEG headband made by Interaxon. A lot of things should work with other EEG setups as well but have not been tried.

## Muse-to-OSC
This script captures the data coming from all the LSL streams and send them in a network using the OSC protocol. Most of the channels are identical to the ones used by [Mind Monitor](https://mind-monitor.com/FAQ.php#Compatibility).

## LSL stream emulator
These scripts stream data stored in XDF format using the LSL protocol. Handy when you want to test things without the headband ;)

## Viewer
This is the Version 2 viewer from https://github.com/alexandrebarachant/muse-lsl, I just modified a few things for convenience.
