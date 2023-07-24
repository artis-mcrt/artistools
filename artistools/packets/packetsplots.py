from pathlib import Path

import matplotlib.pyplot as plt

import artistools as at

CLIGHT = 2.99792458e10
DAY = 86400


def make_2d_packets_plot_imshow(modelpath, timestep_min, timestep_max):
    modeldata, _, vmax_cms = at.inputmodel.get_modeldata_tuple(modelpath)
    em_time = True  # False for arrive time

    hist = at.packets.make_3d_histogram_from_packets(
        modelpath, timestep_min=timestep_min, timestep_max=timestep_max, em_time=em_time
    )

    grid = round(len(modeldata["inputcellid"]) ** (1.0 / 3.0))
    vmax_cms = vmax_cms / CLIGHT

    # # Don't plot empty cells
    # i = 0
    # for z in range(0, grid):
    #     for y in range(0, grid):
    #         for x in range(0, grid):
    #             if modeldata["rho"][i] == 0.0:
    #                 hist[x, y, z] = None
    #             i += 1

    timeminarray = at.get_timestep_times(modelpath=modelpath, loc="start")
    timemaxarray = at.get_timestep_times(modelpath=modelpath, loc="end")
    time_lower = timeminarray[timestep_min]
    time_upper = timemaxarray[timestep_max]
    title = f"{time_lower:.2f} - {time_upper:.2f} days"
    print(f"plotting packets between {title}")
    escapetitle = "pktemissiontime" if em_time else "pktarrivetime"
    title = title + "\n" + escapetitle

    plot_axes_list = ["xz", "xy"]
    for plot_axes in plot_axes_list:
        data, extent = at.plottools.imshow_init_for_artis_grid(grid, vmax_cms, hist / 1e41, plot_axes=plot_axes)

        plt.imshow(data, extent=extent)
        cbar = plt.colorbar()
        # cbar.set_label('n packets', rotation=90)
        cbar.set_label(r"energy emission rate ($10^{41}$ erg/s)", rotation=90)
        # cbar.set_label(r'npackets)', rotation=90)
        plt.xlabel(f"v{plot_axes[0]} / c")
        plt.ylabel(f"v{plot_axes[1]} / c")
        plt.xlim(-vmax_cms, vmax_cms)
        plt.ylim(-vmax_cms, vmax_cms)

        # plt.title(title)
        # plt.show()
        outfilename = f"packets_hist_{time_lower:.2f}d_{plot_axes}_{escapetitle}.pdf"
        plt.savefig(Path(modelpath) / outfilename, format="pdf")
        print(f"Saved {outfilename}")
        plt.clf()


def make_2d_packets_plot_pyvista(modelpath, timestep):
    import pyvista as pv

    modeldata, _, vmax_cms = at.inputmodel.get_modeldata_tuple(modelpath)
    _, x, y, z = at.packets.make_3d_grid(modeldata, vmax_cms)
    mesh = pv.StructuredGrid(x, y, z)

    hist = at.packets.make_3d_histogram_from_packets(modelpath, timestep)

    mesh["energy [erg/s]"] = hist.ravel(order="F")
    # print(max(mesh['energy [erg/s]']))

    sargs = {
        "height": 0.75,
        "vertical": True,
        "position_x": 0.04,
        "position_y": 0.1,
        "title_font_size": 22,
        "label_font_size": 25,
    }

    pv.set_plot_theme("document")  # set white background
    p = pv.Plotter()
    p.set_scale(1.5, 1.5, 1.5)
    single_slice = mesh.slice(normal="y")
    # single_slice = mesh.slice(normal='z')
    p.add_mesh(single_slice, scalar_bar_args=sargs)
    p.show_bounds(
        grid=False,
        xlabel="vx / c",
        ylabel="vy / c",
        zlabel="vz / c",
        ticks="inside",
        minor_ticks=False,
        use_2d=True,
        font_size=26,
        bold=False,
    )
    # labels = dict(xlabel='vx / c', ylabel='vy / c', zlabel='vz / c')
    # p.show_grid(**labels)
    p.camera_position = "zx"
    timeminarray = at.get_timestep_times(modelpath=modelpath, loc="start")
    time = timeminarray[timestep]
    p.add_title(f"{time:.2f} - {timeminarray[timestep + 1]:.2f} days")
    print(pv.global_theme)

    p.show(screenshot=modelpath / f"3Dplot_pktsemitted{time:.1f}days_disk.png")


def plot_packet_mean_emission_velocity(modelpath, write_emission_data=True):
    emission_data = at.packets.get_mean_packet_emission_velocity_per_ts(modelpath)

    plt.plot(emission_data["t_arrive_d"], emission_data["mean_emission_velocity"])

    plt.xlim(0.02, 30)
    plt.ylim(0.15, 0.35)
    plt.xscale("log")
    plt.xlabel("Time (days)")
    plt.ylabel("Mean emission velocity / c")
    plt.legend()

    if write_emission_data:
        emission_data.to_csv(Path(modelpath) / "meanemissionvelocity.txt", sep=" ", index=False)

    outfilename = "meanemissionvelocity.pdf"
    plt.savefig(Path(modelpath) / outfilename, format="pdf")
    print(f"Saved {outfilename}")
