import inspect, torch
from torch_scatter import scatter

def scatter_(name, src, index, dim_size=None):
	if name == 'add': name = 'sum'
	assert name in ['sum', 'mean', 'max']
	out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
	return out[0] if isinstance(out, tuple) else out


class MessagePassing(torch.nn.Module):
	def __init__(self, aggr='add'):
		super(MessagePassing, self).__init__()

		self.message_args = inspect.getargspec(self.message)[0][1:]	# In the defined message function: get the list of arguments as list of string| For eg. in rgcn this will be ['x_j', 'edge_type', 'edge_norm'] (arguments of message function)
		self.update_args  = inspect.getargspec(self.update)[0][2:]	# Same for update function starting from 3rd argument | first=self, second=out

		self.message_args_jump = inspect.getargspec(self.message_jump)[0][1:]

	def propagate(self, aggr, edge_index, **kwargs):
		assert aggr in ['add', 'mean', 'max']
		kwargs['edge_index'] = edge_index


		size = None
		message_args = []

		if kwargs['ts_emb'] != None:

			# get ts_emb
			edge_ts_emb = kwargs['ts_emb']
			node_proj = kwargs['node_proj']
			# check the length
			assert edge_ts_emb.size(0) == edge_index.size(1)
			# concatenate embedding
			for arg in self.message_args:
				if arg[-2:] == '_i':  # If arguments ends with _i then include indic
					tmp = kwargs[arg[:-2]]  # Take the front part of the variable | Mostly it will be 'x',
					size = tmp.size(0)
					node_emb = node_proj(torch.cat((tmp[edge_index[0]],edge_ts_emb),dim=1))  # Lookup for head entities in edges
					message_args.append(node_emb)
				elif arg[-2:] == '_j':
					tmp = kwargs[arg[:-2]]  # tmp = kwargs['x']
					size = tmp.size(0)
					node_emb = node_proj(torch.cat((tmp[edge_index[1]], edge_ts_emb), dim=1))  # Lookup for tail entities in edges
					message_args.append(node_emb)
				else:
					message_args.append(kwargs[arg])
		else:
			for arg in self.message_args:
				if arg[-2:] == '_i':					# If arguments ends with _i then include indic
					tmp  = kwargs[arg[:-2]]				# Take the front part of the variable | Mostly it will be 'x',
					size = tmp.size(0)
					message_args.append(tmp[edge_index[0]])		# Lookup for head entities in edges
				elif arg[-2:] == '_j':
					tmp  = kwargs[arg[:-2]]				# tmp = kwargs['x']
					size = tmp.size(0)
					message_args.append(tmp[edge_index[1]])		# Lookup for tail entities in edges
				else:
					message_args.append(kwargs[arg])		# Take things from kwargs



		update_args = [kwargs[arg] for arg in self.update_args]		# Take update args from kwargs

		out = self.message(*message_args)
		out = scatter_(aggr, out, edge_index[0], dim_size=size)		# Aggregated neighbors for each vertex
		out = self.update(out, *update_args)

		return out

	def message(self, x_j):  # pragma: no cover
		return x_j

	def propagate_jump(self, aggr, edge_index, **kwargs):
		assert aggr in ['add', 'mean', 'max']
		kwargs['edge_index'] = edge_index


		size = None
		message_args = []
		for arg in self.message_args_jump:
			if arg[-2:] == '_i':					# If arguments ends with _i then include indic
				tmp  = kwargs[arg[:-2]]				# Take the front part of the variable | Mostly it will be 'x',
				size = tmp.size(0)
				message_args.append(tmp[edge_index[0]])		# Lookup for head entities in edges
			elif arg[-2:] == '_j':
				tmp  = kwargs[arg[:-2]]				# tmp = kwargs['x']
				size = tmp.size(0)
				message_args.append(tmp[edge_index[1]])		# Lookup for tail entities in edges
			else:
				message_args.append(kwargs[arg])		# Take things from kwargs

		update_args = [kwargs[arg] for arg in self.update_args]		# Take update args from kwargs

		out = self.message_jump(*message_args)
		out = scatter_(aggr, out, edge_index[0], dim_size=size)		# Aggregated neighbors for each vertex
		out = self.update(out, *update_args)

		return out

	def message_jump(self, x_j):  # pragma: no cover
		return x_j

	def update(self, aggr_out):  # pragma: no cover
		return aggr_out
