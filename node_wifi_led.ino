#include <ESP8266WiFi.h>

const char* ssid = "Test";
const char* password = "12345678";

int port = 8080;
int ledPin = 13; // GPIO13
int analogIn = A0; // GPIO13
int ledState = LOW;

WiFiServer server(port);

void setup() {
  Serial.begin(115200);
  delay(10);

  // Variable inits
  ledState = LOW;
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, ledState);

  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  Serial.println("WiFi connected");

  server.begin();
  Serial.println("Server started");

  Serial.print("IP: ");
  Serial.print(WiFi.localIP());
  Serial.print(" Port: ");
  Serial.println(port);
}

void loop() {
  WiFiClient client = server.available();
  if (!client) {
    return;
  }

  Serial.println("new client");
  while(!client.available()) {
    delay(1);
  }

  // Read the first line of the request
  String request = client.readStringUntil('\r');
  Serial.println(request);
  client.flush();

  if (request.indexOf("ON") != -1) {
    ledState = HIGH;
  } else if (request.indexOf("OFF") != -1) {
    ledState = LOW;
  }
  digitalWrite(ledPin, ledState);
  // Si nos interesa devolver un paquete de confirmacion o algo.. client.println('whatevah');
  delay(1);
}
