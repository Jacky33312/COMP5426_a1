CC = mpicc

CFLAGS = -O3

TARGET = gepp_0

SRCS = gepp_0.c

NP = 8

all: $(TARGET) run

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@

run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET)

clean:
	rm -f $(TARGET)
