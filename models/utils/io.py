import torch


def save_model(net, filepath):
	x = net.__class__, net.constructor, net.state_dict()
	torch.save(x, filepath)


def load_model(filepath):
	net_class, net_constructor, net_state_dict = torch.load(filepath, map_location='cpu')
	net = net_class(**net_constructor)
	net.load_state_dict(net_state_dict)
	return net
