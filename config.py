#output sampling frequency
FSAMP = 1000

#step for the segmentation:
T_STEP = 0.25

#SEGMENTATION PARS
WINLEN = 0.25 #
T = 0.5*WINLEN
CI_WIN = 0.05

# t_start = instant of the start of the segment
# t_beat = instant of the nearest next beat

# a segment is considered a "beat" if:
# (T - CI_WIN/2) < (t_beat - t_start) < (T - CI_WIN/2)
