int leds[] = {12,11,9,10,8,7};//{7,8,10,9,11,12}; 

void setup() {
  // put your setup code here, to run once:
  for(int i = 0; i < 8; i++)
    pinMode(7+i, OUTPUT);

  pinMode(13, OUTPUT);
  digitalWrite(13, HIGH);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available())
  {
    int n = ((int)Serial.read());
    n -= 48;
    Serial.println(n);
    for(int i = 0; i < 6; i++)
      digitalWrite(leds[i], i < n ? HIGH : LOW);
    //digitalWrite(leds[n], HIGH);
    delay(100);    
    while(Serial.available())
    {
      delay(100);
      Serial.read();
    }
  }
}
