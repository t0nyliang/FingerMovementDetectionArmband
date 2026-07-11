// Live MLX90393 magnetometer readout through the PCA9548 I2C mux.
// Wiring: ESP32 GPIO21 -> SDA, GPIO22 -> SCL, both to PCA9548 (STEMMA QT).
// MLX90393 is on mux channel 0 at address 0x18.

#include <Wire.h>
#include <Adafruit_MLX90393.h>

constexpr uint8_t MUX_ADDR = 0x70;
constexpr uint8_t MUX_CHANNEL = 0;
constexpr uint8_t SDA_PIN = 21;
constexpr uint8_t SCL_PIN = 22;
constexpr uint8_t MLX90393_ADDR = 0x18;

Adafruit_MLX90393 sensor = Adafruit_MLX90393();
bool sensorFound = false;

void muxSelectChannel(uint8_t channel) {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

bool devicePresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  delay(500);

  Wire.begin(SDA_PIN, SCL_PIN);
  muxSelectChannel(MUX_CHANNEL);

  Serial.println();
  Serial.println(F("MLX90393 Live Readout (via mux channel 0)"));
  Serial.println(F("=========================================="));
}

void loop() {
  muxSelectChannel(MUX_CHANNEL);

  if (!sensorFound) {
    if (devicePresent(MLX90393_ADDR) && sensor.begin_I2C(MLX90393_ADDR, &Wire)) {
      sensorFound = true;
      Serial.println(F("Sensor found! Streaming Bx/By/Bz..."));
    } else {
      Serial.println(F("Sensor not found at 0x18"));
      delay(1000);
      return;
    }
  }

  float x, y, z;
  if (sensor.readData(&x, &y, &z)) {
    // CSV line for the live plotter (plot_live.py) to parse: x,y,z in uT.
    Serial.print(F("DATA,"));
    Serial.print(x, 3);
    Serial.print(',');
    Serial.print(y, 3);
    Serial.print(',');
    Serial.println(z, 3);
    delay(50);
  } else {
    Serial.println(F("Sensor not found at 0x18"));
    sensorFound = false;
    delay(1000);
  }
}
