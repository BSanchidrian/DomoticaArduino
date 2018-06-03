import socket
from threading import Thread
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
from posenet_Learn import posenet_learn

buffer = []
entrenar = False


class Server(WebSocket):
    def handleMessage(self):
        # print(self.address, self.data)  # comentar si no quieres que imprima los datos recibidos
        self.sendMessage('OK')  # para mantener la conexion con la web abierta hay que mandar paquetes cada x tiempo

        global entrenar
        if entrenar and len(buffer) < 10:
            buffer.append(self.data)

        if entrenar and len(buffer) == 10:
            print("Entrenando xd")
            # print(buffer)
            red = posenet_learn(buffer)
            red.fit()
            print(red.predict())
            entrenar = False
            buffer.clear()

            # self.data contiene el json. Ejemplo -> [{"score":0.9941501021385193,"part":"nose","position":{"x":429.9399753965181,"y":251.9409243353482}},{"score":0.8967401385307312,"part":"leftEye","position":{"x":449.82256369755186,"y":229.84977438038794}},{"score":0.9921322464942932,"part":"rightEye","position":{"x":392.2414890683931,"y":224.63147996047448}},{"score":0.12224335223436356,"part":"leftEar","position":{"x":443.4728207292228,"y":266.323702318915}},{"score":0.9927525520324707,"part":"rightEar","position":{"x":288.16951951651737,"y":253.54356347445784}},{"score":0.8558942675590515,"part":"leftShoulder","position":{"x":480.12416886954475,"y":476.50914790712557}},{"score":0.9387611150741577,"part":"rightShoulder","position":{"x":138.06125888166758,"y":465.13222929855874}},{"score":0.06375883519649506,"part":"leftElbow","position":{"x":527.5338267359241,"y":547.161527015423}},{"score":0.08675520122051239,"part":"rightElbow","position":{"x":39.02631377516122,"y":552.3564596241919}},{"score":0.016375452280044556,"part":"leftWrist","position":{"x":453.16153469743404,"y":546.8262290165343}},{"score":0.006576170213520527,"part":"rightWrist","position":{"x":338.95456095728383,"y":285.31840683509563}},{"score":0.004844838287681341,"part":"leftHip","position":{"x":400.8116432979189,"y":429.659282763251}},{"score":0.005912667140364647,"part":"rightHip","position":{"x":66.4027563423946,"y":543.5023466965248}},{"score":0.01751721277832985,"part":"leftKnee","position":{"x":499.6661416941676,"y":510.27799135405445}},{"score":0.006455669179558754,"part":"rightKnee","position":{"x":510.6422790527344,"y":453.75493279818835}},{"score":0.006854082923382521,"part":"leftAnkle","position":{"x":454.56027421622446,"y":508.4433806320717}},{"score":0.003116293577477336,"part":"rightAnkle","position":{"x":515.0302507071659,"y":460.45830151788124}}]
            # resultado = redneuronal.procesar(json)
            # node_mcu.send(resultado)

    def handleConnected(self):
        print(self.address, 'connected')

    def handleClose(self):
        print(self.address, 'closed')


class Client(object):
    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.socket.connect((self.address, self.port))

    def send(self, packet):
        self.socket.send(packet)

    def close(self):
        self.socket.close()


# Como ya no tenemos app de android no deberia hacer falta
# class Server(object):
#     def __init__(self, port):
#         self.port = port
#         self.connection = None
#         self.address = None
#         self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
#     def start(self):
#         self.socket.bind(('192.168.1.41', self.port))
#         self.socket.listen(1)
#
#     def close(self):
#         self.socket.close()
#
#     def wait_connection(self):
#         self.connection, self.address = self.socket.accept()
#         return self.connection

def get_input():
    while True:
        text = input("prompt")
        print(text)
        if text == "y":
            print('Entrenando :D')
            global entrenar
            entrenar = True


if __name__ == '__main__':
    # IP de la placa..
    # node_mcu = Client('192.168.1.114', 8080)
    thread = Thread(target=get_input)
    thread.start()
    websocket_server = SimpleWebSocketServer('', 25000, Server)
    websocket_server.serveforever()
