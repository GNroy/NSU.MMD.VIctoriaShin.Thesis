import numpy as np
class Agent:
    def __init__(self):
        self.location = None
        self.NC_default = None
        self.CFR_default = 0
        self.NC = None
        self.CFR = None
        self.shelter = None
        self.distance = None
        self.contacts = None
    
    def assingn_shelter(self, shelter):
        self.shelter = shelter
        self.distance = self.get_distance(self.shelter)
    
    def flip(self, partner):
        shelter = partner.shelter
        partner.assingn_shelter(self.shelter)
        self.assingn_shelter(shelter)
    
    def set_NC_default(self):
        self.NC = self.NC_default
    
    def set_CFR_default(self):
        self.CFR = self.CFR_default
    
    def ajust_NC(self):
        if self.NC > 0:
            self.NC -= 1
    
    def ajust_CFR(self):
        self.CFR += 1
    
    def init_algo(self, location, contacts):
        self.location = location
        self.NC_default = len(contacts)
        self.set_NC_default()
        self.set_CFR_default()
        self.contacts = contacts
    
    def get_distance(self, shelter):
        return np.linalg.norm(self.location - shelter)
    
    def round(self, order):
        # experiments showed that the following two lines aren't necessary
        #if self.NC > 0:
        #    self.set_NC_default()
        for o in order:
            partner = self.contacts[o]
            s11 = self.distance
            s12 = self.get_distance(partner.shelter)
            s21 = partner.get_distance(self.shelter)
            s22 = partner.distance
            if s11 + s22 > s12 + s21:
                self.flip(partner)
                self.set_NC_default()
                self.set_CFR_default()
                partner.set_NC_default()
                partner.set_CFR_default()
            elif self.NC > 0:
                self.ajust_NC()
                self.set_CFR_default()
                partner.ajust_NC()
                partner.set_CFR_default()
            elif partner.NC == 0:
                self.ajust_CFR()
                partner.ajust_CFR()
            '''
            else:
                self.set_NC_default()
                self.set_CFR_default()
                partner.set_NC_default()
                partner.set_CFR_default()
            '''
