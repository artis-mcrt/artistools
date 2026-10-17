# The physics of ARTIS

This file explains the physical quantities of ARTIS, how ARTIS calculates them, and the papers that describe the
methods. Read it to explain the behaviour of a model. An example is a cell with a high temperature or an unusual
ionisation. SKILL.md gives the readers, and `quantities.md` gives the columns of each file. The "References" section
gives the full reference of each paper.

ARTIS is a code for the 3D, time-dependent radiative transfer in supernovae and kilonovae. It uses the Monte Carlo
method (Sim 2007; Kromer & Sim 2009). The solver of the NLTE populations and of the non-thermal electrons comes from
Shingles et al. (2020).

## Energy packets

ARTIS moves indivisible packets of energy through the cells (Lucy 2002, 2003, 2005). A packet changes its type when
it interacts:

- A **pellet** holds the energy of a radioactive decay. The pellet decays at a sampled time. Then it becomes a
  gamma-ray packet, a fast lepton, an alpha particle, or a fission product.
- A **gamma-ray packet** interacts by Compton scattering, photoelectric absorption, or pair production. The energy
  that it gives to the matter becomes the `deposition_gamma` estimator.
- A **fast lepton** is a non-thermal electron or positron. It gives its energy to heating, ionisation, and excitation
  (see "Non-thermal electrons"). An alpha particle and a fission product give all their energy to heating.
- A **k-packet** is the thermal kinetic energy of the gas. ARTIS selects its next channel with a probability that is
  proportional to the cooling rate of that channel. The channels are free-free, free-bound, and collisional. Thus the
  `cooling_` columns also show where the thermal energy goes.
- A **macro atom** is an ion that holds the energy of a packet in an excited level (Lucy 2002, 2003). An absorption
  or an excitation activates it. The absorption can be bound-bound or bound-free. The excitation can be collisional
  or non-thermal. The macro atom makes internal transitions until it emits an r-packet or it becomes a k-packet.
- An **r-packet** is ultraviolet, optical, or infrared radiation. A free-free absorption makes it a k-packet. A
  bound-bound absorption activates a macro atom. A bound-free absorption activates a macro atom with the probability
  nu_edge / nu. Here, nu is the frequency of the r-packet, and nu_edge is the frequency of the ionisation edge.
  Otherwise, the r-packet becomes a k-packet. An r-packet that leaves the grid adds to the spectra and the light
  curves.
- A **virtual packet** gives the spectrum for one viewing direction with less noise (Bulla, Sim & Kromer 2015). ARTIS
  writes the virtual packets only with the option `VPKT_ON`.

## The radiation field and the temperatures

ARTIS adds the path lengths of the r-packets in a cell to get the mean intensity J and its mean frequency (Kromer &
Sim 2009). At the start of each time step, ARTIS uses J and the mean frequency to calculate these values for each
cell:

- `TJ`, the temperature of a black body with the same J: TJ = (pi J / sigma)^(1/4), with sigma the Stefan-Boltzmann
  constant;
- `TR`, the temperature of a black body with the same mean frequency nu_bar: TR = h nu_bar / (3.832 k), with h the
  Planck constant and k the Boltzmann constant;
- `W`, the dilution factor: W = pi J / (sigma TR^4). A value below 1 shows a radiation field that is weaker than a
  black body at TR, e.g. far from the source of the radiation;
- `Te`, the electron temperature. ARTIS finds the Te at which the heating rate equals the cooling rate. If no Te from
  `MINTEMP` to `MAXTEMP` gives this balance, ARTIS sets Te to one of the two limits. Each step of the solver changes
  Te by a factor of 2 or less.

ARTIS uses LTE in the first time steps. `input.txt` gives the number of these time steps. It also gives the time steps
of the grey approximation. In a time step of the grey approximation, ARTIS uses LTE in a cell that is optically
thick.

In a cell in LTE, Te, TR, and TJ have the same value, and W is 1. ARTIS gets the populations from the Saha equation and
the Boltzmann equation, and it does not solve the thermal balance. Thus a row with W = 1 and Te = TR = TJ shows a cell
in LTE. The heating columns and the cooling columns do not explain the Te of that cell.

With the option `MULTIBIN_RADFIELD_MODEL_ON`, ARTIS also fits a TR and a W to each frequency bin of J (Shingles et al.
2020). From the time step `MULTIBIN_RADFIELD_FIRST_TIMESTEP`, the photoionisation rates use the fit of each bin.

## Ionisation and level populations

ARTIS calculates the ionisation and the level populations of an element in one of two ways:

- **An element with NLTE levels.** ARTIS solves the rate equations of the NLTE levels and the superlevel of each ion
  of the element (Shingles et al. 2020, section 2.3). The rates are the radiative and the collisional rates of the
  bound-bound and the bound-free transitions, and the non-thermal ionisation and excitation.
  `ION_NLEVELS_EXCITED_NLTE` gives the number of excited levels of each ion in the solution.
- **An element with no NLTE levels.** The ratio of two ions comes from the balance of the ionisation and the
  recombination. The ionisation rate is the photoionisation and the collisional ionisation of the ground level, plus
  the non-thermal ionisation. The recombination rate is the radiative recombination at Te.

A **superlevel** holds all the levels of an ion above its NLTE levels. The NLTE solution gives the population of the
superlevel. Inside the superlevel, the level populations follow the Boltzmann distribution at Te. With
`LTEPOP_EXCITATION_USE_TJ`, the Boltzmann distribution uses TJ in place of Te. The levels of an element with no NLTE
levels also use this distribution.

## Non-thermal electrons

The fast leptons of the radioactive decays ionise and excite the ions before they heat the gas. ARTIS solves the
Spencer-Fano equation for the energy distribution of the non-thermal electrons in each cell (Spencer & Fano 1954;
Kozma & Fransson 1992; Shingles et al. 2020). The solution gives the fractions of the deposited energy that go into
heating, ionisation, and excitation. It also gives the non-thermal ionisation rates `gamma_NT_<ion>`. `NT_SCHEME`
(`NT_ON` in older versions of ARTIS) selects this solution. SKILL.md gives an example that compares the non-thermal
ionisation with the photoionisation.

## References

- Bulla M., Sim S. A., Kromer M., 2015, MNRAS, 450, 967-981, doi:10.1093/mnras/stv657, arXiv:1503.07002.
- Kozma C., Fransson C., 1992, ApJ, 390, 602-621, doi:10.1086/171311.
- Kromer M., Sim S. A., 2009, MNRAS, 398, 1809-1826, doi:10.1111/j.1365-2966.2009.15256.x, arXiv:0906.3152.
- Lucy L. B., 2002, A&A, 384, 725-735, doi:10.1051/0004-6361:20011756.
- Lucy L. B., 2003, A&A, 403, 261-275, doi:10.1051/0004-6361:20030357.
- Lucy L. B., 2005, A&A, 429, 19-30, doi:10.1051/0004-6361:20041656.
- Shingles L. J., Sim S. A., Kromer M., Maguire K., Bulla M., Collins C., Ballance C. P., Michel A. S., Ramsbottom
  C. A., Röpke F. K., Seitenzahl I. R., Tyndall N. B., 2020, MNRAS, 492, 2029-2043, doi:10.1093/mnras/stz3412,
  arXiv:1912.02214.
- Sim S. A., 2007, MNRAS, 375, 154-162, doi:10.1111/j.1365-2966.2006.11271.x, arXiv:astro-ph/0611677.
- Spencer L. V., Fano U., 1954, Phys. Rev., 93, 1172-1181, doi:10.1103/PhysRev.93.1172.
