CC = gcc
MPICC = mpicc
CFLAGS = -O3 -std=c99 -g
SDIR=./src
LDIR=./lib
IDIR=./inc
EXE = $(SDIR)/knnring_sequential $(SDIR)/knnring_mpi $(SDIR)/knnring_mpi_syc  $(SDIR)/knnring_mpi_asyc
LIBS = $(LDIR)/knnring_sequential.a $(LDIR)/knnring_mpi.a $(LDIR)/knnring_mpi_syc.a $(LDIR)/knnring_mpi_asyc.a
MAIN = tester


all: $(EXE)
lib: $(LIBS)



$(SDIR)/knnring_sequential: $(SDIR)/$(MAIN).c $(LDIR)/knnring_sequential.a
	$(CC) -I$(IDIR) -L$(LDIR) -o $@ $^ -lopenblas -lm

$(SDIR)/knnring_mpi: $(SDIR)/$(MAIN)_mpi.c $(LDIR)/knnring_mpi.a
	$(MPICC) -I$(IDIR) -L$(LDIR) -o $@ $^ -lopenblas -lm

$(SDIR)/knnring_mpi_syc: $(SDIR)/$(MAIN)_mpi.c $(LDIR)/knnring_mpi_syc.a
	$(MPICC) -I$(IDIR) -L$(LDIR) -o $@ $^ -lopenblas -lm

$(SDIR)/knnring_mpi_asyc: $(SDIR)/$(MAIN)_mpi.c $(LDIR)/knnring_mpi_asyc.a
	$(MPICC) -I$(IDIR) -L$(LDIR) -o $@ $^ -lopenblas -lm




$(LDIR)/%.a: $(SDIR)/%.o $(SDIR)/utilities.o
	ar rcs $@ $^




$(SDIR)/knnring_sequential.o:	$(SDIR)/knnring_sequential.c
	$(CC)  -I$(IDIR) $(CFLAGS) -L$(LDIR) -o $@ -c $< -lm

$(SDIR)/knnring_mpi.o:	$(SDIR)/knnring_mpi.c
	$(MPICC) -I$(IDIR) $(CFLAGS) -L$(LDIR) -o $@ -c $< -lm

$(SDIR)/knnring_mpi_syc.o:	$(SDIR)/knnring_mpi_syc.c
	$(MPICC) -I$(IDIR) $(CFLAGS) -L$(LDIR) -o $@ -c $< -lm

$(SDIR)/knnring_mpi_asyc.o:	$(SDIR)/knnring_mpi_asyc.c
	$(MPICC) -I$(IDIR) $(CFLAGS) -L$(LDIR) -o $@ -c $< -lm

$(SDIR)/utilities.o:	$(SDIR)/utilities.c
	$(MPICC) -I$(IDIR) $(CFLAGS) -L$(LDIR) -o $@ -c $< -lm



clean:
	rm -f $(SDIR)/*.o $(EXE) $(LIBS)
