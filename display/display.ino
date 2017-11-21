#include <Adafruit_GFX.h>
#include <Adafruit_PCD8544.h>
Adafruit_PCD8544 display = Adafruit_PCD8544(3, 4, 5, 6, 7);
long numbers = 0;
const int timeRes = 43;
byte buff[timeRes];

int serReadInt()
{
 int i, serAva;                           // i is a counter, serAva hold number of serial available
 char inputBytes [7];                 // Array hold input bytes
 char * inputBytesPtr = &inputBytes[0];  // Pointer to the first element of the array
     
 if (Serial.available()>0)            // Check to see if there are any serial input
 {
   delay(5);                              // Delay for terminal to finish transmitted
                                              // 5mS work great for 9600 baud (increase this number for slower baud)
   serAva = Serial.available();  // Read number of input bytes
   for (i=0; i<serAva; i++)       // Load input bytes into array
     inputBytes[i] = Serial.read();
   inputBytes[i] =  '\0';             // Put NULL character at the end
   return atoi(inputBytesPtr);    // Call atoi function and return result
 }
 else
   return -1;                           // Return -1 if there is no input
}

void setup() {
    Serial.begin(9600);
  
    // инициализация и очистка дисплея
    display.begin();
    display.clearDisplay();
    display.display();
    
    display.setContrast(50); // установка контраста
    delay(1000);
    display.setTextSize(1);  // установка размера шрифта
    display.setTextColor(BLACK); // установка цвета текста
    display.setCursor(0,0); // установка позиции курсора
  
    display.println("Connect me!");
    display.display();

    for(int i = 0; i < timeRes; i++)
      buff[i] = 0;
}

void loop() {
  String msg;
  
  if(Serial.available())
  {
    int x = serReadInt();
    buff[numbers%timeRes] = x;
    display.clearDisplay();
    
    display.drawLine(((numbers%timeRes)+1)*2-2, 0, ((numbers%timeRes)+1)*2-2, 48, BLACK);
    for(int i = 0; i < timeRes; i++)
    {     
      if(i != 0 && i != (numbers%timeRes)+1 && i != (numbers%timeRes)+2)
        display.drawLine(i*2-2, 48-buff[i-1], i*2, 48-buff[i], BLACK);
        display.drawLine(i*2-2, 48-buff[i-1]+1, i*2, 48-buff[i]+1, BLACK);
      //display.drawPixel(i*2, 48-buff[i], BLACK);
      //display.drawPixel(i*2, 48-(buff[i]+1), BLACK);
    }
    display.display();
  
    numbers++;
  }

    

   
}
