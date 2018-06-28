module mod_NNTools
use mod_Precision
use mod_Log
implicit none

    contains

	!* 计算 L_2误差
    subroutine calc_L_2_error( t, y, err )
    implicit none
        !* t 是实际输出，y 是网络预测输出
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(in) :: y
        real(PRECISION), intent(inout) :: err
        
        integer :: t_shape(2)  
                
        t_shape = SHAPE(t)      
		
		!* root mean square error.                  
        err = SUM((t - y)*(t - y))
        err = err / ( t_shape(1) * t_shape(2) )
        err = SQRT(err)

        call LogDebug("mod_NNTools: SUBROUTINE calc_L2_error.")
             
        return
    end subroutine calc_L_2_error
    !====  
	
	!* 计算 L_∞误差
    subroutine calc_L_inf_error( t, y, err )
    implicit none
        !* t 是实际输出，y 是网络预测输出
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(in) :: y
        real(PRECISION), intent(inout) :: err    
		                
        err = MAXVAL(ABS(t - y))

        call LogDebug("mod_NNTools: SUBROUTINE calc_L_inf_error.")
             
        return
    end subroutine calc_L_inf_error
    !====  
	
	!* 用交叉熵计算误差
    subroutine calc_cross_entropy_error( t, y, err, max_err )
    implicit none
        !* t 是实际输出，y 是网络预测输出
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(in) :: y
        real(PRECISION), optional, intent(inout) :: err   
		real(PRECISION), optional, intent(out) :: max_err

		real(PRECISION) :: local_err, local_max_err
		integer :: t_shape(2)
		integer :: i, j
		real(PRECISION) :: tmp
		
		if ((PRESENT(err) == .false.) .and. &
			(PRESENT(max_err) == .false.)) then
			call LogErr("mod_NNTools: SUBROUTINE calc_cross_entropy_error, &
				dummy variable err or max_err both inexistence.")
			stop
		end if
		       
		t_shape = SHAPE(t)
			   
		local_err = 0
		local_max_err = 0
					 
        do j=1, t_shape(2)
			tmp = 0
			!* error = -DOT_PRODUCT(t(:,j), LOG(y(:,j)))
			do i=1, t_shape(1)              
				if (abs(t(i,j)) < 1.E-16 .and. abs(y(i,j)) < 1.E-16) then
					!* 定义 0*log(0) = 0 
					continue
				else
					tmp = tmp - t(i,j) * LOG(y(i,j))
				end if
			end do
			
			if (tmp > local_max_err)  local_max_err = tmp
			
			local_err = local_err + tmp
			
		end do
  
		local_err = local_err / t_shape(2)

		if (PRESENT(err))  err = local_err
		
		if (PRESENT(max_err))  max_err = local_max_err
		
        call LogDebug("mod_NNTools: SUBROUTINE calc_cross_entropy_error.")
             
        return
    end subroutine calc_cross_entropy_error
    !====  
	
	!* 正确率计算
	!*   分类问题：需要根据网络输出计算正确率.
    subroutine calc_classify_accuracy( t, y, acc )
    implicit none
		!* t 是实际输出，y 是网络预测输出
		!* 这里假设 t 是 one-hot 编码，而 y 是 softmax 的输出
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(in) :: y
        real(PRECISION), intent(inout) :: acc
    
        integer :: y_shape(2), j, tag
        integer :: max_index_t(1), max_index_y(1)
        
        y_shape = SHAPE(y)
        
        tag = 0
        do j=1, y_shape(2)
            max_index_t = MAXLOC(t(:,j))
            max_index_y = MAXLOC(y(:,j))
            
            if (max_index_t(1) == max_index_y(1)) then
                tag = tag + 1
            end if
        end do
        
        acc = 1.0 * tag / y_shape(2)
        
        call LogDebug("mod_NNTools: SUBROUTINE calc_classify_accuracy.")
        
        return
    end subroutine calc_classify_accuracy
 
end module mod_NNTools
