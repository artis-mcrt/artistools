#!/usr/bin/env python3
import math
import numpy as np
from astropy import units as u
from astropy import constants as c

t = 2.56004e+07 * u.s
comeanlife = 111.3 * u.day
mconucleus = 55.9398393 * u.u
mco_0 = 0.641305 * u.solMass
v_outer = 8000 * u.km / u.s
volume = (4 * math.pi / 3 * (v_outer * t) ** 3).to('cm3')

print(f't_now = {t.to("d")}')
print(f'volume = {volume}')
mco_now = mco_0 * math.exp(-t / comeanlife)
print(f'Mco_now = {mco_now}')
power = (4.566 * u.MeV * (mco_now / mconucleus) / comeanlife).to('erg/s')
power = (0.610 * 0.19 * u.MeV * (mco_now / mconucleus) / comeanlife).to('erg/s')
print(f'epsilon = {power / volume}')