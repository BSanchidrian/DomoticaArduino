## Setup

Instalar las dependencias:

```python
sudo pip install git+https://github.com/dpallot/simple-websocket-server.git
```

y hay que cambiar la ip de la placa NodeMCU

```python
node_mcu = Client('<ip_placa_nodemcu>', 8080)
```