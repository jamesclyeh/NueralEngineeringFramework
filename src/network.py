from collections import defaultdict
import copy

"""
Network object containing clusters and connections between them
"""
class Network:
  def __init__(self):
    self.nodes = {}
    self.node_source = defaultdict(set)
    self.node_destination = defaultdict(set)
    self.root_nodes = set()

  def RemoveNode(self, node):
    del nodes[node.name]
    del node_source[node.name]
    for key, val in node_source.iteritems():
      if node.name in val:
        val.remove(node.name)
    del node_destination[node.name]
    if node.name in root_nodes:
      root_nodes.remove(node.name)

  """Adding connection between neuron populations and/or input/output
  """
  def AddConnection(
      self, source, destination, transformation=lambda x: x):
    self.nodes[source.name] = source
    self.nodes[destination.name] = destination
    if source.name not in self.node_source:
      self.root_nodes.add(source.name)
    self.node_source[destination.name].add(source.name)
    self.node_destination[source.name].add(destination.name)
    self.nodes[destination.name].GenerateDecoders(transformation)
    self.root_nodes = self.root_nodes.difference(destination.name)

  """Get the order to execute the connections so that dependencies of
  connections will be satisfied prior to execution
  """
  def GetExecutionOrder(self):
    bottom_level_nodes = copy.deepcopy(self.root_nodes)
    execution_order = []
    while len(bottom_level_nodes) != 0:
      to_add = bottom_level_nodes.pop()
      execution_order.append(to_add)
      for n in self.node_destination[to_add]:
        if self.node_source[n].issubset(set(execution_order)):
          bottom_level_nodes.add(n)
    return execution_order

  def GetNetworkOutput(self):
    execution_order = self.GetExecutionOrder()
    for node_name in execution_order:
      node = self.nodes[node_name]
      if self.node_destination[node_name]:
        for destination_name in self.node_destination[node_name]:
          self.nodes[destination_name].SetInput(node.GetOutput())

    return self.nodes[execution_order[-1]].GetOutput()
