module mod_ShuffleBatchGenerator
use mod_BaseRandomBatchGenerator
use mod_Log
use mod_Precision
implicit none

!---------------------
! 工作类：无放回抽样 |
!---------------------
type, extends(BaseRandomBatch), public :: ShuffleBatchGenerator
    !* 继承自BaseRandomBatch并实现其接口

    logical, private :: is_random_init = .false.	
	logical, private :: is_shuffle_done = .false.
	
	integer, private :: index_lower = 0, index_up = 0
    
!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: get_next_batch => m_get_next_batch

end type ShuffleBatchGenerator
!===================

 
    !-------------------------
    private :: m_get_next_batch
	private :: m_random_int
	private :: m_shuffle
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||
       

	!* 获取一个batch
	subroutine m_get_next_batch( this, X_train, y_train, X_batch, y_batch ) 
	use	mod_Tools
	implicit none
		class(ShuffleBatchGenerator), intent(inout) :: this
		real(PRECISION), dimension(:,:), intent(inout) :: X_train
		real(PRECISION), dimension(:,:), intent(inout) :: y_train
		real(PRECISION), dimension(:,:), intent(out) :: X_batch
        real(PRECISION), dimension(:,:), intent(out) :: y_batch

		integer :: X_train_shape(2), X_batch_shape(2)
		integer :: i, j
		
		associate (                            &
            index_lower => this % index_lower, &
            index_up    => this % index_up     &
        )   	
		
        X_train_shape = SHAPE(X_train)
		X_batch_shape = SHAPE(X_batch)
		
        if (this % is_random_init == .false.) then
            call RANDOM_SEED()
            this % is_random_init = .true.
        end if
        
		if (this % is_shuffle_done == .false.) then
			call m_shuffle( X_train_shape(2), X_train, y_train )
			this % is_shuffle_done = .true.
		end if
		
		index_lower = index_up + 1
		index_up    = index_up + X_batch_shape(2)
		
		if (index_up <= X_train_shape(2)) then
		
			X_batch = X_train(:, index_lower:index_up)
			y_batch = y_train(:, index_lower:index_up)
			
        else if (index_lower <= X_train_shape(2)) then	
     			
			j = X_train_shape(2) - index_lower + 1			
			X_batch(:, 1:j) = X_train(:, index_lower:X_train_shape(2))
			y_batch(:, 1:j) = y_train(:, index_lower:X_train_shape(2))

			call m_shuffle( X_train_shape(2), X_train, y_train )
			
			index_up = index_up - X_train_shape(2)		
			X_batch(:, j+1:X_batch_shape(2)) = X_train(:, 1:index_up)
			y_batch(:, j+1:X_batch_shape(2)) = y_train(:, 1:index_up)
            
        else if (index_lower > X_train_shape(2)) then	
            
            call m_shuffle( X_train_shape(2), X_train, y_train )
		
			index_lower = index_lower - X_train_shape(2)
			index_up    = index_up - X_train_shape(2)
		
			X_batch = X_train(:, index_lower:index_up)
			y_batch = y_train(:, index_lower:index_up)
			
		else
			call LogErr("ShuffleBatchGenerator: SUBROUTINE m_get_next_batch.")
		end if
		
		end associate
		
		call LogDebug("ShuffleBatchGenerator: SUBROUTINE m_get_next_batch.")
		
		return
	end subroutine
	!====
	
	!* 静态函数，随机生成一个在[M,N]范围的整数.
	subroutine m_random_int( M, N, ans )
    implicit none
        integer, intent(in)  :: M, N
        integer, intent(out) :: ans

        real(PRECISION) :: tmp

        if( M > N ) call LogErr("c_randInt: M > N")

        call RANDOM_NUMBER( tmp )

        ans = FLOOR( tmp * ( M - N + 1 ) + N )

        return
    end subroutine m_random_int
    
	!* 随机“洗牌”
	subroutine m_shuffle( need_size, X_train, y_train )
    implicit none
        integer, intent(in) :: need_size
        real(PRECISION), dimension(:,:), intent(inout) :: X_train
		real(PRECISION), dimension(:,:), intent(inout) :: y_train
		
        integer :: X_train_shape(2), y_train_shape(2)
		integer :: i, j, step_lower
        real(PRECISION), dimension(:), allocatable :: X_tmp, y_tmp

		X_train_shape = SHAPE(X_train)	
		y_train_shape = SHAPE(y_train)	

		allocate( X_tmp(X_train_shape(1)) )
		allocate( y_tmp(y_train_shape(1)) )
		
        step_lower = X_train_shape(2) - need_size + 1
        do i=X_train_shape(2), step_lower, -1
		
			call m_random_int(1, i, j)
			
			X_tmp = X_train(:, i)
			y_tmp = y_train(:, i)
			
			X_train(:, i) = X_train(:, j)
			y_train(:, i) = y_train(:, j)
			
			y_train(:, j) = X_tmp
			y_train(:, j) = y_tmp
			
		end do
		
		deallocate( X_tmp )
		deallocate( y_tmp )

        return
    end subroutine m_shuffle
	
	
end module