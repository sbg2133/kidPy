from lab_brick import core


class Attenuator():
    def __init__(self, attenuator_type, address):
        if attenuator_type == 'lab_brick':
            self.instr = lab_brick_init(address)
            self.get_fn = self.instr.get_attenuation
            self.set_fn = self.instr.set_attenuation
        else:
            raise NotImplementedError('Not an implemented attenuator type')

    def get_atten(self):
        return self.get_fn()

    def set_atten(self, atten_val):
        return self.set_fn(atten_val)


def lab_brick_init(address):
    return core.Attenuator(0x041f, 0x1208, address)
