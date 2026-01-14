import math
import nav
from connection.ComputerReceiver import ComputerReceiver

from . import protocol


class CommandSender:
    def __init__(command_client_socket):
        self.computerReceiver = computerReceiver
