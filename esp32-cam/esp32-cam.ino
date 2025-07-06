#include "esp_camera.h"
#include <WiFi.h>
#include <AsyncUDP.h>
#include "esp_timer.h"
#include "esp_log.h"
//
// WARNING!!! PSRAM IC required for UXGA resolution and high JPEG quality
//            Ensure ESP32 Wrover Module or other board with PSRAM is selected
//            Partial images will be transmitted if image exceeds buffer size
//
//            You must select partition scheme from the board menu that has at least 3MB APP space.
//            Face Recognition is DISABLED for ESP32 and ESP32-S2, because it takes up from 15
//            seconds to process single frame. Face Detection is ENABLED if PSRAM is enabled as well

// ===================
// Select camera model
// ===================
//#define CAMERA_MODEL_WROVER_KIT // Has PSRAM
//#define CAMERA_MODEL_ESP_EYE  // Has PSRAM
//#define CAMERA_MODEL_ESP32S3_EYE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_PSRAM // Has PSRAM
//#define CAMERA_MODEL_M5STACK_V2_PSRAM // M5Camera version B Has PSRAM
//#define CAMERA_MODEL_M5STACK_WIDE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_ESP32CAM // No PSRAM
//#define CAMERA_MODEL_M5STACK_UNITCAM // No PSRAM
//#define CAMERA_MODEL_M5STACK_CAMS3_UNIT  // Has PSRAM
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
//#define CAMERA_MODEL_TTGO_T_JOURNAL // No PSRAM
//#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM
// ** Espressif Internal Boards **
//#define CAMERA_MODEL_ESP32_CAM_BOARD
//#define CAMERA_MODEL_ESP32S2_CAM_BOARD
//#define CAMERA_MODEL_ESP32S3_CAM_LCD
//#define CAMERA_MODEL_DFRobot_FireBeetle2_ESP32S3 // Has PSRAM
//#define CAMERA_MODEL_DFRobot_Romeo_ESP32S3 // Has PSRAM
#include "camera_pins.h"

// ===========================
// Enter your WiFi credentials
// ===========================
const char *ssid = "Reuth 1";
const char *password = "followthewhiterabbit";

static const char* TAG = "CAMERA_DISCOVERY";

AsyncUDP udp;

void setupDiscovery() {
    // Start UDP listener on port 8888
    if(udp.listen(8888)) {
        Serial.println("UDP discovery listener started on port 8888");
        ESP_LOGI(TAG, "UDP discovery listener started on port 8888");
        
        // Set up packet handler for discovery requests
        udp.onPacket([](AsyncUDPPacket packet) {
            String request = String((char*)packet.data());
            request.trim(); // Remove any whitespace
            
            ESP_LOGI(TAG, "Received discovery request: '%s' from %s", request.c_str(), packet.remoteIP().toString().c_str());
            Serial.printf("Received discovery request: '%s' from %s\n", request.c_str(), packet.remoteIP().toString().c_str());
            
            // Check if this is a discovery request
            if (request == "DISCOVER_CAMERAS") {
                // Send response with camera info
                String response = "ESP32_CAMERA:" + WiFi.localIP().toString() + ":80";
                
                // Create AsyncUDPMessage and send response
                AsyncUDPMessage message(response.length());
                message.write((uint8_t*)response.c_str(), response.length());
                udp.sendTo(message, packet.remoteIP(), 8888);
                
                ESP_LOGI(TAG, "Sent discovery response: %s to %s", response.c_str(), packet.remoteIP().toString().c_str());
                Serial.printf("Sent discovery response: %s to %s\n", response.c_str(), packet.remoteIP().toString().c_str());
            }
        });
    } else {
        ESP_LOGE(TAG, "Failed to start UDP discovery listener");
        Serial.println("Failed to start UDP discovery listener");
    }
}

void startCameraServer();
void setupLedFlash(int pin);

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 24000000;  // Increased for better performance
  config.frame_size = FRAMESIZE_SVGA;  // 800x600 for higher quality
  config.pixel_format = PIXFORMAT_JPEG;  // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_LATEST;  // Always get the latest frame
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 8;  // Higher quality (lower number = better quality)
  config.fb_count = 3;  // More frame buffers for smoother streaming

  // if PSRAM IC present, init with optimized settings for streaming
  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.jpeg_quality = 8;  // High quality for PSRAM
      config.fb_count = 3;  // More buffers for smoother streaming
      config.grab_mode = CAMERA_GRAB_LATEST;  // Always get latest frame
    } else {
      // Limit the frame size when PSRAM is not available
      config.frame_size = FRAMESIZE_VGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
      config.jpeg_quality = 12;  // Better quality even for DRAM
    }
  } else {
    // Best option for face detection/recognition
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // flip it back
    s->set_brightness(s, 1);   // up the brightness just a bit
    s->set_saturation(s, -2);  // lower the saturation
  }
  // Optimize for high quality streaming
  if (config.pixel_format == PIXFORMAT_JPEG) {
    s->set_framesize(s, FRAMESIZE_SVGA);  // 800x600 for high quality
    s->set_quality(s, 8);  // High quality (lower number = better quality)
    s->set_brightness(s, 1);  // Slightly increased brightness
    s->set_contrast(s, 1);   // Slightly increased contrast
    s->set_saturation(s, 1); // Slightly increased saturation
    s->set_special_effect(s, 0); // No special effects
    s->set_whitebal(s, 1);   // Enable white balance
    s->set_awb_gain(s, 1);   // Enable AWB gain
    s->set_wb_mode(s, 0);    // Auto white balance
    s->set_exposure_ctrl(s, 1); // Enable exposure control
    s->set_aec2(s, 1);       // Enable AEC2 for better exposure
    s->set_gain_ctrl(s, 1);  // Enable gain control
    s->set_agc_gain(s, 0);   // Default AGC gain
    s->set_gainceiling(s, (gainceiling_t)2); // Higher gain ceiling for better low light
    s->set_bpc(s, 1);        // Enable BPC for better quality
    s->set_wpc(s, 1);        // Enable WPC
    s->set_raw_gma(s, 1);    // Enable raw gamma
    s->set_lenc(s, 1);       // Enable lens correction
    s->set_hmirror(s, 0);    // No horizontal mirror
    s->set_vflip(s, 0);      // No vertical flip
    s->set_dcw(s, 1);        // Enable DCW
    s->set_colorbar(s, 0);   // Disable colorbar
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

// Setup LED FLash if LED pin is defined in camera_pins.h
#if defined(LED_GPIO_NUM)
  setupLedFlash(LED_GPIO_NUM);
#endif

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  startCameraServer();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");

  // Start UDP for discovery
    setupDiscovery();
}

void loop() {
  // Do nothing. Everything is done in another task by the web server
  delay(10000);
}