from dm_control.mjcf import parser
import json
import numpy as np


def randomize_doublecartpole_geometry(mjcf):
    """
    Takes in a dmcontrol mjcf object and,
    Modifies,
    1. The geometries: gpole1, gpole2, and gmass to desired values
    2. The locations of the body coordinate frames of pole1, pole2, and mass for consistency
    :param mjcf: dmcontrol mjcf object
    :return:
    """
    # Find random lengths for pole1 and pole2 within a reasonable range
    pole1_len, pole2_len = np.random.uniform(low=0.3, high=0.7, size=2)
    # Find the same random width for pole1 and pole2 within a reasonable range
    #  width is parameterized by the radius of the hemi-spherical capsule caps
    pole_width = np.random.uniform(low=0.05, high=0.15)
    # Find random radius for the mass within a reasonable range, should be larger than
    #  the width of the pole
    mass_rad = np.random.uniform(low=pole_width + 0.03, high=0.2)

    # Create a dictionary out of above random params
    dcpole_params = {}
    dcpole_params["pole1"] = pole1_len
    dcpole_params["pole2"] = pole2_len
    dcpole_params["pole_width"] = pole_width
    dcpole_params["mass_radius"] = mass_rad

    fname = "gym_cenvs/assets/dcpole.json"
    with open(fname, 'w') as f:
        f.write(json.dumps(dcpole_params))

    # Retrieve pointers to bodies
    pole1 = mjcf.worldbody.body['cart'].body['pole1']
    pole2 = pole1.body['pole2']
    mass = pole2.body['mass']

    # Adjust the length of gpole1
    pole1.geom['gpole1'].fromto = [0, 0, 0, 0, 0, pole1_len]
    # Set radius of capsule caps on pole1 as half of desired pole_width
    pole1.geom['gpole1'].size = [pole_width]
    # Adjust the location of the body frame of pole2 so that gpole2 and hinge2 start from the right position
    pole2.pos = [0, 0, pole1_len]
    # Adjust the length of gpole2
    pole2.geom['gpole2'].fromto = [0, 0, 0, 0, 0, pole2_len]
    # Set radius of capsule caps on pole2 as half of desired pole_width
    pole2.geom['gpole2'].size = [pole_width]
    # Adjust the body location of mass so that its geom is centered correct at the end of the pole
    mass.pos = [0, 0, pole2_len]
    # Modify the radius of the mass
    mass.geom['gmass'].size = [mass_rad, mass_rad]
    return


def main():
    """
    Read in doublecartpole_static.xml
    Write modified/new doublecartpole_dynamic.xml file to disk
    :return:
    """
    # Read in existing dcartpole static xml file using dmc parser
    mjcf = parser.from_path("gym_cenvs/assets/doublecartpole_static.xml")

    # Modify mjcf model parameters in place
    randomize_doublecartpole_geometry(mjcf)

    # Send modified mjcf model to string
    doublecartpole_dynamic_str = mjcf.to_xml_string()

    # Write string to dynamic variant xml file
    with open("gym_cenvs/assets/doublecartpole_dynamic.xml", "w") as f:
        f.write(doublecartpole_dynamic_str)


if __name__ == '__main__':
    main()
