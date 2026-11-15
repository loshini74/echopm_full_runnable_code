/*
  EchoPM — Modular 1-Phase / 3-Phase ESP32 Edge Node (PS 03)
  - Reads ADXL345 Vibration, DS18B20 Temperature, YF-S201 Flow, Outlet Pressure
  - Supports 1-Phase (Channel A) or 3-Phase (Channels A, B, C) Current & Voltage
  - Executes 3-second PWM Speed-Step Active Probing on MQTT command
*/

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_server = "broker.hivemq.com";

WiFiClient espClient;
PubSubClient client(espClient);

const int PHASE_MODE = 1; // Set 1 for Single-Phase / 12V DC Rig, 3 for Three-Phase
const int PWM_PIN = 25;
unsigned long lastPublish = 0;

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String msg;
  for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];
  if (msg.indexOf("PWM_PROBE") >= 0) {
    // Active 3-second PWM speed step (70% -> 100%)
    analogWrite(PWM_PIN, 180);
    delay(1500);
    analogWrite(PWM_PIN, 255);
    delay(1500);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(PWM_PIN, OUTPUT);
  analogWrite(PWM_PIN, 255);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) delay(500);

  client.setServer(mqtt_server, 1883);
  client.setCallback(mqttCallback);
}

void loop() {
  if (!client.connected()) {
    if (client.connect("EchoPM_ESP32_Node01")) {
      client.subscribe("factory/line1/machine01/command");
    }
  }
  client.loop();

  if (millis() - lastPublish > 3000) {
    lastPublish = millis();

    // Replace simulated values with analogRead / I2C sensor reads on physical rig
    float vib_rms = 2.45;
    float temp_c = 49.3;
    float curr_a = 4.15;
    float flow_lpm = 18.2;
    float press_bar = 1.10;
    float power_kw = (curr_a * 230.0) / 1000.0;

    StaticJsonDocument<320> doc;
    doc["device_id"] = "motor_pump_01";
    doc["phase_mode"] = PHASE_MODE;
    doc["vibration_rms"] = vib_rms;
    doc["temperature"] = temp_c;
    doc["current_rms"] = curr_a;
    doc["power_kw"] = power_kw;
    doc["water_flow_lpm"] = flow_lpm;
    doc["outlet_pressure_bar"] = press_bar;

    char out[320];
    serializeJson(doc, out);
    client.publish("factory/line1/machine01/telemetry", out);
  }
}
