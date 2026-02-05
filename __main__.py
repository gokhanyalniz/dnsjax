from parameters import I_START, T_START
import vfield
import transform

def dns():

    it = I_START
    t = T_START

    # Start from the laminar state to test
    vfieldk_now = vfield.get_laminar()
    vfieldx_now = transform.spec_to_phys_vector(vfieldk_now)
    return

if __name__ == '__main__':
    dns()