module mod_SimpleBatchGenerator
use mod_BaseRandomBatchGenerator
use mod_Log
use mod_Precision
implicit none

!---------------------
! 工作类：有放回抽样 |
!---------------------
type, extends(BaseRandomBatch), public :: SimpleBatchGenerator
    !* 继承自BaseRandomBatch并实现其接口

    logical, private :: is_random_init = .false.
    
!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: get_next_batch => m_get_next_batch

end type SimpleBatchGenerator
!===================

 
    !-------------------------
    private :: m_get_next_batch
	private :: m_random_int
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||
       

	!* 获取一个batch
	subroutine m_get_next_batch( this, X_train, y_train, X_batch, y_batch )   
	implicit none
		class(SimpleBatchGenerator), intent(inout) :: this
		real(PRECISION), dimension(:,:), intent(in) :: X_train
		real(PRECISION), dimension(:,:), intent(in) :: y_train
		real(PRECISION), dimension(:,:), intent(out) :: X_batch
        real(PRECISION), dimension(:,:), intent(out) :: y_batch

		integer :: X_train_shape(2), X_batch_shape(2)
		integer :: train_count, batch_count
		integer :: j, random_index
		
		X_train_shape = SHAPE(X_train)
		X_batch_shape = SHAPE(X_batch)
		
		train_count = X_train_shape(2)
		batch_count = X_batch_shape(2)
		
		if (this % is_random_init == .false.) then
            call RANDOM_SEED()
            this % is_random_init = .true.
        end if
        
		do j=1, batch_count
			call m_random_int(1, train_count, random_index)
			X_batch(:, j) = X_train(:, random_index)
			y_batch(:, j) = y_train(:, random_index)
		end do
		
		call LogDebug("SimpleBatchGenerator: SUBROUTINE m_get_next_batch.")
		
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

        call random_number( tmp )

        ans = floor( tmp * ( M - N + 1 ) + N )

        return
    end subroutine m_random_int
    
end module