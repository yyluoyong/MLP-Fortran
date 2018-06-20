module mod_PReLU
use mod_Precision
use mod_BaseActivationFunction
implicit none    

	!* undo: ���ļ��ж�ȡ PReLU �Ĳ���
	real(PRECISION), parameter :: PReLU_ALPHA = 0.1

!-------------------
! �����ࣺ����� |
!-------------------
type, extends(BaseActivationFunction), public :: PReLU
    !* �̳���BaseActivationFunction��ʵ����ӿ�

!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: f       => m_fun_PReLU
    procedure, public :: f_vect  => m_fun_PReLU_vect 
    procedure, public :: df      => m_df_PReLU
    procedure, public :: df_vect => m_df_PReLU_vect

end type PReLU
!===================

    !-------------------------
    private :: m_fun_PReLU
    private :: m_df_PReLU
    private :: m_fun_PReLU_vect
	private :: m_df_PReLU_vect
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* PReLU����
    subroutine m_fun_PReLU( this, index, x, y )
    implicit none
        class(PReLU), intent(inout) :: this
        integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: y
    
        if (x(index) < 0) then
            y = PReLU_ALPHA * x(index) 
        else
            y = x(index)
        end if
    
        return
    end subroutine
    !====
    
    !* �������������PReLU����
	subroutine m_fun_PReLU_vect( this, x, y )
	implicit none
        class(PReLU), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: y
	
		integer :: count, i
	
        count = SIZE(X)
        
        do i=1, count
            if (x(i) < 0) then
                y(i) = PReLU_ALPHA * x(i)
            else
                y(i) = x(i)
            end if
        end do
        
		return
	end subroutine
	!====
    
	!* PReLU������һ�׵���
	subroutine m_df_PReLU( this, index, x, dy )
	implicit none
        class(PReLU), intent(inout) :: this
		integer, intent(in) :: index
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), intent(out) :: dy
	
		if (x(index) < 0) then
            dy = PReLU_ALPHA
        else
            dy = 1
        end if
	
		return
	end subroutine
	!====
	
	!* �������������PReLU������һ�׵���
	subroutine m_df_PReLU_vect( this, x, dy )
	implicit none
        class(PReLU), intent(inout) :: this
		real(PRECISION), dimension(:), intent(in) :: x
		real(PRECISION), dimension(:), intent(out) :: dy
	
		integer :: count, i
	
        count = SIZE(X)
        
        do i=1, count
            if (x(i) < 0) then
                dy(i) = PReLU_ALPHA
            else
                dy(i) = 1
            end if
        end do
	
		return
	end subroutine
	!====

end module