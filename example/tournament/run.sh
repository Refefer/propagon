# Computes the power rankings for baseball's 2018 season
# This assumes you have the propagon binary compiled and installed!
propagon dehydrate baseball.2018

# Run the example with different model types
propagon baseball.2018.edges glicko2 > baseball.2018.glicko2
propagon baseball.2018.glicko2 hydrate --vocab baseball.2018.vocab > scores.glicko2

propagon baseball.2018.edges btm-mm > baseball.2018.btm-mm
propagon baseball.2018.btm-mm hydrate --vocab baseball.2018.vocab > scores.btm-mm

propagon baseball.2018.edges btm-lr > baseball.2018.btm-lr
propagon baseball.2018.btm-lr hydrate --vocab baseball.2018.vocab > scores.btm-lr

propagon baseball.2018.edges rate > baseball.2018.rate
propagon baseball.2018.rate hydrate --vocab baseball.2018.vocab > scores.rate

