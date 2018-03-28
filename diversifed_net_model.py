import torch


# Conv.
class ConvLayer(torch.nn.Module):
    def __init__(self, inc, outc, kW, kH, sW=None, sH=None, mW=None, mH=None):
        super(ConvLayer, self).__init__()

        # Set the Stride if it is not set to be 1.
        csW = sW if sW is not None else 1
        csH = sH if sH is not None else 1

        # Compute the Current Padding.
        cmW = mW if mW is not None else 1
        cmH = mH if mH is not None else 1
        
        cpW = (kW - 1) // 2 * cmW
        cpH = (kH - 1) // 2 * cmH
        
        # Create the Convolutional Layer.
        self.conv_layer = torch.nn.Conv2d(inc, outc, (kW, kH), stride=(csW, csH), padding=(cpW, cpH))

    def forward(self, X):
        return self.conv_layer(X)

# Layer Set Input.
class LayerSetInput(torch.nn.Module):
    def __init__(self, inc, outc):
        super(LayerSetInput, self).__init__()        
        self.layer_conv_tranpose = torch.nn.ConvTranspose2d(inc, outc, 8, 8)
        self.layer_normalization = torch.nn.InstanceNorm2d(outc)
        self.layer_leaky_relu = torch.nn.LeakyReLU(0, True)


    def forward(self, X):
        current_output = self.layer_conv_tranpose(X)
        current_output = self.layer_normalization(current_output)
        current_output = self.layer_leaky_relu(current_output)    
        return current_output

# Layer Set A.
class LayerSetA(torch.nn.Module):
    def __init__(self, input_channels):
        super(LayerSetA, self).__init__()
        self.layer_upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.layer_normalization = torch.nn.InstanceNorm2d(input_channels)

        self.layer_a_conv = ConvLayer(input_channels, input_channels, 3, 3)
        self.layer_a_norm = torch.nn.InstanceNorm2d(input_channels)
        self.layer_a_relu = torch.nn.LeakyReLU(0, True)

        self.layer_b_conv = ConvLayer(input_channels, input_channels, 3, 3)
        self.layer_b_norm = torch.nn.InstanceNorm2d(input_channels)
        self.layer_b_relu = torch.nn.LeakyReLU(0, True)

        self.layer_c_conv = ConvLayer(input_channels, input_channels, 1, 1)
        self.layer_c_norm = torch.nn.InstanceNorm2d(input_channels)
        self.layer_c_relu = torch.nn.LeakyReLU(0, True)


    def forward(self, X):
        output = self.layer_normalization(self.layer_upsample(X))
        output = self.layer_a_relu(self.layer_a_norm(self.layer_a_conv(output)))
        output = self.layer_b_relu(self.layer_b_norm(self.layer_b_conv(output)))
        output = self.layer_c_relu(self.layer_c_norm(self.layer_c_conv(output)))
        return output


# Layer Set B.
class LayerSetB(torch.nn.Module):
    def __init__(self, input_channels):
        super(LayerSetB, self).__init__()     

        self.layer_upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.layer_normalization = torch.nn.InstanceNorm2d(input_channels)

        self.layer_a_conv = ConvLayer(input_channels, input_channels//2, 3, 3)
        self.layer_a_norm = torch.nn.InstanceNorm2d(input_channels//2)
        self.layer_a_relu = torch.nn.LeakyReLU(0, True)

        self.layer_b_conv = ConvLayer(input_channels//2, input_channels//2, 3, 3)
        self.layer_b_norm = torch.nn.InstanceNorm2d(input_channels//2)
        self.layer_b_relu = torch.nn.LeakyReLU(0, True)

        self.layer_c_conv = ConvLayer(input_channels//2, input_channels//2, 1, 1)
        self.layer_c_norm = torch.nn.InstanceNorm2d(input_channels//2)
        self.layer_c_relu = torch.nn.LeakyReLU(0, True)


    def forward(self, X):
        output = self.layer_normalization(self.layer_upsample(X))
        output = self.layer_a_relu(self.layer_a_norm(self.layer_a_conv(output)))
        output = self.layer_b_relu(self.layer_b_norm(self.layer_b_conv(output)))
        output = self.layer_c_relu(self.layer_c_norm(self.layer_c_conv(output)))        
        return output


# Layer Set Output.
class LayerSetOutput(torch.nn.Module):
    def __init__(self, inc, outc):
        super(LayerSetOutput, self).__init__()
        self.layer_conv = ConvLayer(inc, outc, 3, 3)

    def forward(self, X):
        return self.layer_conv(X)





# Diversified Net.
class DiversifiedNet(torch.nn.Module):
    def __init__(self, inc, conv_channels, out_channels):
        super(DiversifiedNet, self).__init__()

        # The Layers of the Diversified Network.
        self.layer_input = LayerSetInput(inc, conv_channels)

        self.layer_set_a_1 = LayerSetA(conv_channels)
        self.layer_set_b_1 = LayerSetB(conv_channels)

        self.layer_set_a_2 = LayerSetA(conv_channels // 2)
        self.layer_set_b_2 = LayerSetB(conv_channels // 2)

        self.layer_set_b_3 = LayerSetB(conv_channels // 4)

        self.layer_output = LayerSetOutput(conv_channels // 8, out_channels)


    def forward(self, X):

        output = self.layer_input(X)

        output = self.layer_set_a_1(output)
        output = self.layer_set_b_1(output)

        output = self.layer_set_a_2(output)
        output = self.layer_set_b_2(output)

        output = self.layer_set_b_3(output)

        output = self.layer_output(output)

        return output





        





