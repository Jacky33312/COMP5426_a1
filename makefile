CC = gcc

CFLAGS = -fopenmp -O3

TARGET = gepp_0

SRCS = gepp_0.c


all: $(TARGET)


$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@


clean:
	rm -f $(TARGET)
