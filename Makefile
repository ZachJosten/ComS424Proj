CC = mpicc
FFLAGS = -Wall -std=c99
LFLAGS =
OBJECTS = main.o matrix.o

main.exe: $(OBJECTS)
	$(CC) $(LFLAGS) $(OBJECTS) -o main.exe -lm

%.o: %.c
	$(CC) $(FFLAGS) -c $< -lm

clean:
	rm -f $(OBJECTS) *.exe