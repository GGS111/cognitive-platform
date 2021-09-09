import asyncio
import socket

async def recv_data(c_socket):
    while True : # also add an interuption logic as break the loop if empty string or what you have there
        data = await loop.sock_recv(c_socket, 2048)
        if data == '':
            break
        
        print(data)





def create_socket(port, host='localhost'):
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind((host, port))
    server.setblocking(False)
    return server

print('start')
async def main():
    UDP_IP = "127.0.0.1"
    RAW_PORT = 125
    MCU_PORT = 126
    SERVER_PORT = 5001

    server = create_socket(SERVER_PORT, UDP_IP)

    listen_connection = loop.create_task(recv_data(server))

    await asyncio.gather(listen_connection)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()







# import socket

# UDP_IP = "127.0.0.1"
# UDP_PORT = 5001

# sock = socket.socket(socket.AF_INET, # Internet
#                      socket.SOCK_DGRAM) # UDP
# sock.bind((UDP_IP, UDP_PORT))

# while True:
#     data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
#     print("received message: %s" % data)