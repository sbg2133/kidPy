#!/usr/bin/env python3

import socket

def ETCPClient(tcpIP, tcpPort):
    BUFFER_SIZE = 10

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((tcpIP, tcpPort))
    data = s.recv(BUFFER_SIZE)
    s.close()

    ux = int(data[0:5])
    uy = int(data[5:10])

    return ux, uy

def main():
    vx, vy = ETCPClient('137.79.43.78', 30245)
    print (vx, vy)

if __name__ == "__main__":
    main()
