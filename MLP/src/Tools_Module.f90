module mod_Tools
use mod_Precision
implicit none

    contains

    !* 输出Tecplot曲线
    !* 这里x是自变量，即横轴，var_i是因变量，即纵轴.
    !* 其中，var_2至var_9是可选的，即最多可以输出9条曲线.
    subroutine output_tecplot_line( file_name, &
        x_name, x,                             &
        var_1_name, var_1, var_2_name, var_2,  &
        var_3_name, var_3, var_4_name, var_4,  &
        var_5_name, var_5, var_6_name, var_6,  &
        var_7_name, var_7, var_8_name, var_8,  &
        var_9_name, var_9 )
    implicit none
        character(len=*), intent(in) :: file_name
        character(len=*), intent(in) :: x_name, var_1_name
        real(PRECISION),  dimension(:), intent(in) :: x, var_1
        character(len=*), optional, intent(in) :: &
            var_2_name, var_3_name, var_4_name, &
            var_5_name, var_6_name, var_7_name, &
            var_8_name, var_9_name
        real(PRECISION), dimension(:), optional, intent(in) :: &
            var_2, var_3, var_4, var_5,  &
            var_6, var_7, var_8, var_9

        character(len=300) :: var_name_list 
        integer :: var_count, i
        
        var_count = SIZE(x)
        
        var_name_list = 'variables="' // TRIM(ADJUSTL(x_name)) // &
            '","' // TRIM(ADJUSTL(var_1_name)) // '"'
        
        if (PRESENT(var_2_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_2_name)) // '"' 
        end if
        if (PRESENT(var_3_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_3_name)) // '"'
        end if
        if (PRESENT(var_4_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_4_name)) // '"'
        end if
        if (PRESENT(var_5_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_5_name)) // '"'
        end if
        if (PRESENT(var_6_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_6_name)) // '"'
        end if
        if (PRESENT(var_7_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_7_name)) // '"'
        end if
        if (PRESENT(var_8_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_8_name)) // '"'
        end if
        if (PRESENT(var_9_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_9_name)) // '"'
        end if
        
        open(unit=33, file=file_name, form='formatted', status='replace')
        
        write(33, *) 'title="By SUBROUTINE output_tecplot_line"'    
        write(33, *) TRIM(ADJUSTL(var_name_list))
        
        do i=1, var_count
            write(33, '(E15.7, E15.7\)') x(i), var_1(i)
        
            if (PRESENT(var_2))  write(33, '(E15.7\)') var_2(i)
            if (PRESENT(var_3))  write(33, '(E15.7\)') var_3(i)
            if (PRESENT(var_4))  write(33, '(E15.7\)') var_4(i)
            if (PRESENT(var_5))  write(33, '(E15.7\)') var_5(i)
            if (PRESENT(var_6))  write(33, '(E15.7\)') var_6(i)
            if (PRESENT(var_7))  write(33, '(E15.7\)') var_7(i)
            if (PRESENT(var_8))  write(33, '(E15.7\)') var_8(i)
            if (PRESENT(var_9))  write(33, '(E15.7\)') var_9(i)

            write(33, *)
        end do
        
        close(33)
        
        return
    end subroutine output_tecplot_line
    !====
    
    !* 输出Tecplot二维云图数据
    !* 这里 X 是 X 坐标，Y 是 Y 坐标.
    !* 其中，var_2 至 var_9 是可选的，即最多可以输出9个云图.
    subroutine output_tecplot_2D( file_name, &
        X_name, X, Y_name, Y,  &
        var_1_name, var_1, var_2_name, var_2,  &
        var_3_name, var_3, var_4_name, var_4,  &
        var_5_name, var_5, var_6_name, var_6,  &
        var_7_name, var_7, var_8_name, var_8,  &
        var_9_name, var_9 )
    implicit none
        character(len=*), intent(in) :: file_name
        character(len=*), intent(in) :: X_name, Y_name, var_1_name
        real(PRECISION),  dimension(:,:), intent(in) :: X, Y, var_1
        character(len=*), optional, intent(in) :: &
            var_2_name, var_3_name,               &
            var_4_name, var_5_name, var_6_name,   &
            var_7_name, var_8_name, var_9_name
        real(PRECISION), dimension(:,:), optional, intent(in) :: &
            var_2, var_3, var_4, var_5,  &
            var_6, var_7, var_8, var_9

        character(len=300) :: var_name_list 
        character(len=100) :: var_count_list
        character(len=20) :: i_count_to_string, j_count_to_string 
        integer :: var_shape(2), i, j
        
        var_shape = SHAPE(X)
        
        var_name_list = 'variables="' // TRIM(ADJUSTL(X_name)) // &
            '","' // TRIM(ADJUSTL(Y_name)) // '","' //            &
            TRIM(ADJUSTL(var_1_name)) // '"'
        
        if (PRESENT(var_2_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_2_name)) // '"'
        end if
        if (PRESENT(var_3_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_3_name)) // '"'
        end if
        if (PRESENT(var_4_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_4_name)) // '"'
        end if
        if (PRESENT(var_5_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_5_name)) // '"'
        end if
        if (PRESENT(var_6_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_6_name)) // '"'
        end if
        if (PRESENT(var_7_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_7_name)) // '"'
        end if
        if (PRESENT(var_8_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_8_name)) // '"'
        end if
        if (PRESENT(var_9_name)) then
            var_name_list = TRIM(ADJUSTL(var_name_list)) // ',"' // &
                TRIM(ADJUSTL(var_9_name)) // '"'
        end if
        
        write(i_count_to_string, *)  var_shape(1)
        write(j_count_to_string, *)  var_shape(2)
        
        var_count_list = 'zone i = ' // &
            TRIM(ADJUSTL(i_count_to_string)) // ', j = ' // &
            TRIM(ADJUSTL(j_count_to_string))
        
        open(unit=33, file=file_name, form='formatted', status='replace')
        
        write(33, *) 'title="By SUBROUTINE output_tecplot_line"'    
        write(33, *) TRIM(ADJUSTL(var_name_list))
        write(33, *) TRIM(ADJUSTL(var_count_list))
    
        do j=1, var_shape(2)
            do i=1, var_shape(1) 
                write(33, '(E15.7, E15.7, E15.7\)') &
                    X(i, j), Y(i, j), var_1(i, j)
           
                if (PRESENT(var_2))  write(33, '(E15.7\)') var_2(i, j)
                if (PRESENT(var_3))  write(33, '(E15.7\)') var_3(i, j)
                if (PRESENT(var_4))  write(33, '(E15.7\)') var_4(i, j)
                if (PRESENT(var_5))  write(33, '(E15.7\)') var_5(i, j)
                if (PRESENT(var_6))  write(33, '(E15.7\)') var_6(i, j)
                if (PRESENT(var_7))  write(33, '(E15.7\)') var_7(i, j)
                if (PRESENT(var_8))  write(33, '(E15.7\)') var_8(i, j)
                if (PRESENT(var_9))  write(33, '(E15.7\)') var_9(i, j)
            
                write(33, *)
            end do
        end do
        
        return
    end subroutine output_tecplot_2D
    !====
    
    
    subroutine test_output_tecplot_line()
    implicit none
        real(PRECISION), dimension(:), allocatable :: &
            X, Y_sin, Y_cos, Y_sqrt, Y_sinh, Y_tanh, Y_exp
            
        integer :: x_count = 401, i
        real(PRECISION) :: dx
        
        allocate( X(x_count) )
        allocate( Y_sin(x_count) )
        allocate( Y_cos(x_count) )
        allocate( Y_sqrt(x_count) )
        allocate( Y_sinh(x_count) )
        allocate( Y_tanh(x_count) )
        allocate( Y_exp(x_count) )
        
        dx = 0.01
        do i=1, x_count
            X(i) = -2 + dx * (i - 1)
            Y_sin(i) = SIN(X(i))
            Y_cos(i) = COS(X(i))
            Y_sqrt(i) = SQRT(ABS(X(i)))
            Y_sinh(i) = SINH(X(i))
            Y_tanh(i) = TANH(X(i))
            Y_exp(i) = EXP(X(i))
        end do
        
        call output_tecplot_line(                         &
            './Output/TEST/test_output_tecplot_line.plt', &
            'X', X, 'Y_sin', Y_sin, 'Y_cos', Y_cos,       &
            'Y_sqrt', Y_sqrt, 'Y_sinh', Y_sinh,           &
            'Y_tanh', Y_tanh, 'Y_exp', Y_exp )
    
        deallocate( X )
        deallocate( Y_sin )
        deallocate( Y_cos )
        deallocate( Y_sqrt )
        deallocate( Y_sinh )
        deallocate( Y_tanh )
        deallocate( Y_exp )
        
        return
    end subroutine test_output_tecplot_line
    
    
    subroutine test_output_tecplot_2D()
    implicit none
        real(PRECISION), dimension(:, :), allocatable :: &
            X, Y, Z_sin, Z_cos, Z_sqrt, Z_sinh, Z_tanh, Z_exp
            
        integer :: x_count = 401, y_count = 401
        integer :: i, j
        real(PRECISION) :: dx, dy
        
        allocate( X(x_count, y_count) )
        allocate( Y(x_count, y_count) )
        allocate( Z_sin(x_count, y_count) )
        allocate( Z_cos(x_count, y_count) )
        allocate( Z_sqrt(x_count, y_count) )
        allocate( Z_sinh(x_count, y_count) )
        allocate( Z_tanh(x_count, y_count) )
        allocate( Z_exp(x_count, y_count) )
        
        dx = 0.01
        dy = 0.01
        do j=1, y_count
            do i=1, x_count
                X(i, j) = dx * (i - 1)
                Y(i, j) = dy * (j - 1)
                Z_sin(i, j) = SIN(X(i,j) + Y(i,j))
                Z_cos(i, j) = COS(X(i,j) + Y(i,j))
                Z_sqrt(i, j) = SQRT(ABS(X(i,j) + Y(i,j)))
                Z_sinh(i, j) = SINH(X(i,j) + Y(i,j))
                Z_tanh(i, j) = TANH(X(i,j) + Y(i,j))
                Z_exp(i, j) = EXP(X(i,j) + Y(i,j))
            end do  
        end do
        
        call output_tecplot_2D(                         &
            './Output/TEST/test_output_tecplot_2D.plt', &
            'X', X, 'Y', Y, 'Z_sin', Z_sin,             & 
            'Z_cos', Z_cos, 'Z_sqrt', Z_sqrt,           & 
            'Z_sinh', Z_sinh, 'Z_tanh', Z_tanh,         & 
            'Z_exp', Z_exp )
    
        deallocate( X )
        deallocate( Y )
        deallocate( Z_sin )
        deallocate( Z_cos )
        deallocate( Z_sqrt )
        deallocate( Z_sinh )
        deallocate( Z_tanh )
        deallocate( Z_exp )
        
        return
    end subroutine test_output_tecplot_2D
 
end module mod_Tools
