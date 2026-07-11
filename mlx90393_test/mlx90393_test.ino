// Direct-connection test for Adafruit MLX90393 magnetometer (no mux).
// Wiring: SDA -> GPIO21, SCL -> GPIO22, VIN -> 3.3V, GND -> GND.

#include <Wire.h>
#include <Adafruit_MLX90393.h>

constexpr uint8_t SDA_PIN = 21;
constexpr uint8_t SCL_PIN = 22;
constexpr uint8_t MLX90393_ADDR = 0x18;

Adafruit_MLX90393 sensor = Adafruit_MLX90393();
bool sensorFound = false;

// Cheap presence check that avoids calling the library's begin_I2C()
// repeatedly while no device is present -- doing that corrupts the heap
// (Adafruit_MLX90393 leaks/reinitializes its I2C device object on each
// failed begin_I2C() call).
bool devicePresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  delay(500);

  Wire.begin(SDA_PIN, SCL_PIN);

  Serial.println();
  Serial.println(F("MLX90393 Direct Connection Test"));
  Serial.println(F("================================"));
}

void loop() {
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
    Serial.print(F("Bx: "));
    Serial.print(x);
    Serial.print(F(" uT  By: "));
    Serial.print(y);
    Serial.print(F(" uT  Bz: "));
    Serial.print(z);
    Serial.println(F(" uT"));
    delay(500);
  } else {
    Serial.println(F("Sensor not found at 0x18"));
    sensorFound = false;
    delay(1000);
  }
}
