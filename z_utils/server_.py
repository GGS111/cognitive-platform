import socket
import json

# HOST, PORT = "172.16.119.4", 13000
HOST, PORT = "127.0.0.1", 5001
# data = " ".join(sys.argv[1:])
data = 'dfgdgdf'

map_ = {'id': 1, 't': {'x': 0.0, 'y': 0.0, 'z': 0.0}}

data = json.dumps(map_)
# SOCK_DGRAM is the socket type to use for UDP sockets
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# As you can see, there is no connect() call; UDP has no connections.
# Instead, data is directly sent to the recipient via sendto().
sock.sendto(bytes(data + "\n", "utf-8"), (HOST, PORT))
# received = str(sock.recv(1024), "utf-8")

print("Sent:     {}".format(data))