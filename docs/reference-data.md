# Reference data

A plot can show an observed spectrum or an observed light curve beside the models. artistools holds some of these
files in `artistools/data/refspectra` and `artistools/data/lightcurves`.

## Reference spectra

Give the file name of a reference spectrum as a positional argument of `plotspectra`, beside the model paths:

```sh
artistools plotspectra -t 300 mymodel 2003du_20031213_3219_8822_00.txt
```

artistools looks for the file in the working folder first. Then it looks in the reference spectra of the package.
artistools also accepts a compressed file, e.g. with the extension `.zst`, `.gz`, or `.xz`.

A reference spectrum is a text file with two columns or more: the wavelength in Å, then the flux density f_λ. A line
that starts with `#` is a comment. The file `<name>.meta.yml` beside it gives the metadata. These keys change the
plot:

| Key | Effect |
| --- | --- |
| `label` | The label of the spectrum in the legend. |
| `dist_mpc` | The distance in Mpc. The command scales the flux to the distance of the plot. Without `--distmpc`, the plot takes the `dist_mpc` of the first reference spectrum, or 1 Mpc. Without this key, the command takes the flux to be at the distance of the plot, and it gives a warning. |
| `t` | The time of the spectrum in days. `--scaletoreftime` uses it. |
| `z` | The redshift. The command changes the spectrum to the rest frame. Without this key, the command takes z = 0 and gives a warning. |
| `a_v`, `r_v`, `e_bminusv` | The extinction. Two of these keys give the third. The command removes the reddening with the law of Cardelli, Clayton & Mathis (1989, ApJ, 345, 245-256, doi:10.1086/167900). |
| `mask_telluric` | `true` removes the telluric bands at 13500 to 14400 Å and 18000 to 19400 Å. |
| `scale_factor` | A factor for the flux. |
| `f_lambda_columnindex` | The index of the flux column, from 0 for the first column. The default is 1, which is the second column. |

Some metadata files in the package have no data file, because the data is not public. To use such a spectrum, put
the data file and a copy of its metadata file in the working folder. artistools reads the metadata file beside the
data file that it finds.

## Reference light curves

Give the file names of reference light curves with `--reflightcurves` of `plotlightcurves`. `plotlightcurves --help`
gives the other options of the bands and the magnitudes.
