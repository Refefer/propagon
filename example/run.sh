# Computes the power rankings for baseball's 2018 season
# This assumes you have the propagon binary compiled and installed!
python ../scripts/remap.py baseball.2018

# Run the example with different model types
propagon baseball.2018.remap glicko2 > baseball.2018.glicko2
python ../scripts/unmap.py baseball.2018.ids baseball.2018.glicko2 > scores.glicko2

propagon baseball.2018.remap btm-mm > baseball.2018.btm-mm
python ../scripts/unmap.py baseball.2018.ids baseball.2018.btm-mm > scores.btm-mm

propagon baseball.2018.remap btm-lr > baseball.2018.btm-lr
python ../scripts/unmap.py baseball.2018.ids baseball.2018.btm-lr > scores.btm-lr

propagon baseball.2018.remap rate > baseball.2018.rate
python ../scripts/unmap.py baseball.2018.ids baseball.2018.rate > scores.rate

