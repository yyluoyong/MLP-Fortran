PROGRAM MAIN
use mod_Log
use mod_Precision
use mod_NNTrain
use mod_MNISTCase
use mod_SinCase
implicit none
    
    type(MNISTCase), pointer :: my_MNISTCase
    type(SinCase), pointer :: my_SinCase
    
    allocate( my_SinCase ) 
    
    call my_SinCase % main()

    !real(PRECISION), dimension(:,:), allocatable :: X, t, y
    !integer, parameter :: SAMPLE_COUNT = 201
    !type(NNTrain), pointer :: my_NNTrain
    !
    !integer :: i
    !real(PRECISION) :: PI, dx
    !
    !allocate(X(1,SAMPLE_COUNT))
    !allocate(t(1,SAMPLE_COUNT))
    !allocate(y(1,SAMPLE_COUNT))
    !allocate(my_NNTrain)
    !
    !PI = 4 * ATAN(1.0)
    !
    !dx = 2.0 / (SAMPLE_COUNT - 1)
    !do i=1, SAMPLE_COUNT
    !    X(1, i) = -1 + (i-1) * dx
    !    t(1, i) = SIN(X(1, i))
    !end do
    !
    !t = 0.5 * (t + 1)
    
    !X(1,1) = 0.1
    !X(1,2) = 0.15
    !X(1,3) = 1.1
    !X(1,4) = 0.88
    !
    !t(1,1) = 0.2
    !t(1,2) = 0.9
    !t(1,3) = 0.01
    !t(1,4) = 0.95
    
    
    !call my_NNTrain % train(X, t, y)
    
END PROGRAM