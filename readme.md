[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Muse tools
This repository contains scripts to work with the Muse2 EEG headband made by Interaxon. A lot of things should work with other EEG setups as well but have not been tried.

## Muse-to-OSC
This script captures the data coming from all the LSL streams and send them in a network using the OSC protocol. Most of the channels are identical to the ones used by [Mind Monitor](https://mind-monitor.com/FAQ.php#Compatibility).

## LSL stream emulator
These scripts stream data stored in XDF format using the LSL protocol. Handy when you want to test things without the headband ;)
The icon comes from https://commons.m.wikimedia.org/wiki/File:Assessment_brain_icon.png

## Viewer
This is the Version 2 viewer from https://github.com/alexandrebarachant/muse-lsl, I just modified a few things for convenience.


:warning: All of the data (except the raw signals) are re-computed and *do not come from the official Muse SDK*. Although the calculations are not very heavy, it's not optimal, but it does have the advantage of 1/ being very easy to use 2/ being customizable.