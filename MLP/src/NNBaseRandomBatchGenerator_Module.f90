module mod_BaseRandomBatchGenerator
implicit none
    
!-----------------------
! �����ࣺ�����ȡ���� |
!-----------------------
type, abstract, public :: BaseRandomBatch

!||||||||||||    
contains   !|
!||||||||||||

    procedure(abs_get_next_batch), deferred, public :: get_next_batch 

end type BaseRandomBatch
!===================
    

!-------------------
! �����ࣺ�����ӿ� |
!-------------------	
abstract interface   

	!* ��ȡһ��batch
	subroutine abs_get_next_batch( this, X_train, y_train, X_batch, y_batch )
    use mod_Precision
    import :: BaseRandomBatch
	implicit none
		class(BaseRandomBatch), intent(inout) :: this
		real(PRECISION), dimension(:,:), intent(in) :: X_train
		real(PRECISION), dimension(:,:), intent(in) :: y_train
		real(PRECISION), dimension(:,:), intent(out) :: X_batch
        real(PRECISION), dimension(:,:), intent(out) :: y_batch

	end subroutine
	!====
	

end interface
!===================
    
end module