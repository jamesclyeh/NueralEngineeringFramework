from src.network import Network
from src.utils import FrozenBunch

import unittest

class NetworkTest(unittest.TestCase):
  def setUp(self):
    self.NodeA = FrozenBunch(name='A')
    self.NodeB = FrozenBunch(name='B')
    self.NodeC = FrozenBunch(name='C')
    self.NodeD = FrozenBunch(name='D')
    self.NodeE = FrozenBunch(name='E')
    self.NodeF = FrozenBunch(name='F')
    self.NodeG = FrozenBunch(name='G')
    self.NodeH = FrozenBunch(name='H')

  """Test GetExecutionOrder dependency sorting algorithm
  """
  def testGetExecutionOrder(self):
    network = Network()
    network.AddConnection(self.NodeA, self.NodeB)
    network.AddConnection(self.NodeA, self.NodeD)
    network.AddConnection(self.NodeB, self.NodeC)
    network.AddConnection(self.NodeC, self.NodeE)
    network.AddConnection(self.NodeC, self.NodeD)
    network.AddConnection(self.NodeD, self.NodeE)
    self.assertEquals(
        [self.NodeA.name, self.NodeB.name, self.NodeC.name,
            self.NodeD.name, self.NodeE.name],
        network.GetExecutionOrder())

    network2 = Network()
    network2.AddConnection(self.NodeA, self.NodeC)
    network2.AddConnection(self.NodeA, self.NodeD)
    network2.AddConnection(self.NodeB, self.NodeD)
    network2.AddConnection(self.NodeC, self.NodeE)
    network2.AddConnection(self.NodeE, self.NodeH)
    network2.AddConnection(self.NodeF, self.NodeE)
    network2.AddConnection(self.NodeF, self.NodeG)
    network2.AddConnection(self.NodeG, self.NodeH)
    self.assertEquals(
        [self.NodeA.name, self.NodeC.name, self.NodeB.name,
            self.NodeD.name, self.NodeF.name, self.NodeE.name,
            self.NodeG.name, self.NodeH.name],
        network2.GetExecutionOrder())

if __name__ == '__main__':
  unittest.main()
