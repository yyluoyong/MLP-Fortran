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
	implicit none
		class(ShuffleBatchGenerator), intent(inout) :: this
		real(PRECISION), dimension(:,:), intent(inout) :: X_train
		real(PRECISION), dimension(:,:), intent(inout) :: y_train
		real(PRECISION), dimension(:,:), intent(out) :: X_batch
        real(PRECISION), dimension(:,:), intent(out) :: y_batch

		integer :: X_train_shape(2), X_batch_shape(2)
        integer :: lower_index
        
        X_train_shape = SHAPE(X_train)
		X_batch_shape = SHAPE(X_batch)
		
        if (this % is_random_init == .false.) then
            call RANDOM_SEED()
            this % is_random_init = .true.
        end if
        
		call m_shuffle( X_train, y_train )
		
        lower_index = X_train_shape(2) - X_batch_shape(2) + 1
		X_batch = X_train(:, lower_index:X_train_shape(2))
		y_batch = y_train(:, lower_index:X_train_shape(2))
		
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
	subroutine m_shuffle( X_train, y_train )
    implicit none
        real(PRECISION), dimension(:,:), intent(inout) :: X_train
		real(PRECISION), dimension(:,:), intent(inout) :: y_train
		
        integer :: X_train_shape(2), y_train_shape(2)
		integer :: i, j
        real(PRECISION), dimension(:), allocatable :: X_tmp, y_tmp

		X_train_shape = SHAPE(X_train)	
		y_train_shape = SHAPE(y_train)	

		allocate( X_tmp(X_train_shape(1)) )
		allocate( y_tmp(y_train_shape(1)) )
		
        do i=X_train_shape(2), 2, -1
		
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