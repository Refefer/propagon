# Computes the power rankings for baseball's 2018 season
# This assumes you have the pairwizer binary compiled and installed!
python ../scripts/remap.py baseball.2018

# Run the example with different model types
pairwizer baseball.2018.remap glicko2 > baseball.2018.glicko2
pairwizer baseball.2018.remap btm-mm > baseball.2018.btm
pairwizer baseball.2018.remap rate > baseball.2018.rate

